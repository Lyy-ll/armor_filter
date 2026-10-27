#include "extended_kalman_filter.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Measurement {
    double x;
    double y;
    double z;
    double yaw;
};

struct FilterConfig {
    // 过程噪声 Q
    double q_position = 1e-5;
    double q_yaw = 1e-6;
    double q_velocity = 1e-3;
    double q_yaw_velocity = 1e-4;

    // 观测噪声 R
    double r_position = 2e-4;
    double r_yaw = 5e-3;

    // yaw 离群值门限，单位为 rad
    double yaw_gate = 1.0;

    // 数据间隔相同，直接按每帧 dt = 1 处理
    double dt = 1.0;
};

static double normalizeAngle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

static std::vector<Measurement> readCsv(const std::string& path)
{
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    std::vector<Measurement> measurements;
    std::string line;

    // 跳过表头
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream stream(line);
        std::string cell;
        std::vector<double> values;

        while (std::getline(stream, cell, ',')) {
            values.push_back(std::stod(cell));
        }

        if (values.size() != 4) {
            throw std::runtime_error("Invalid CSV line: " + line);
        }

        measurements.push_back({
            values[0],
            values[1],
            values[2],
            values[3]
        });
    }

    return measurements;
}

static Eigen::MatrixXd makeTransitionMatrix(double dt)
{
    /*
     * 状态：
     * [x, y, z, yaw, vx, vy, vz, vyaw]
     */
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(8, 8);

    F(0, 4) = dt;
    F(1, 5) = dt;
    F(2, 6) = dt;
    F(3, 7) = dt;

    return F;
}

static Eigen::MatrixXd makeObservationMatrix()
{
    /*
     * 观测：
     * [x, y, z, yaw]
     */
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 8);

    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;
    H(3, 3) = 1.0;

    return H;
}

static Eigen::MatrixXd makeProcessNoise(const FilterConfig& config)
{
    Eigen::VectorXd q(8);

    q <<
        config.q_position,
        config.q_position,
        config.q_position,
        config.q_yaw,
        config.q_velocity,
        config.q_velocity,
        config.q_velocity,
        config.q_yaw_velocity;

    return q.asDiagonal();
}

static Eigen::MatrixXd makeMeasurementNoise(const FilterConfig& config)
{
    Eigen::VectorXd r(4);

    r <<
        config.r_position,
        config.r_position,
        config.r_position,
        config.r_yaw;

    return r.asDiagonal();
}

static void writeCsv(
    const std::string& path,
    const std::vector<Eigen::VectorXd>& filtered_states)
{
    std::ofstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    file << "filtered_x,filtered_y,filtered_z,filtered_yaw\n";

    for (const auto& state : filtered_states) {
        file << std::fixed << std::setprecision(12)
             << state(0) << ','
             << state(1) << ','
             << state(2) << ','
             << normalizeAngle(state(3)) << '\n';
    }
}

static void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program << " input.csv output.csv\n"
        << "  " << program
        << " input.csv output.csv"
        << " q_position q_yaw q_velocity q_yaw_velocity"
        << " r_position r_yaw yaw_gate\n";
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 10) {
        printUsage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    FilterConfig config;

    // 可通过命令行修改 Q、R 和 yaw 门限
    if (argc == 10) {
        config.q_position = std::stod(argv[3]);
        config.q_yaw = std::stod(argv[4]);
        config.q_velocity = std::stod(argv[5]);
        config.q_yaw_velocity = std::stod(argv[6]);
        config.r_position = std::stod(argv[7]);
        config.r_yaw = std::stod(argv[8]);
        config.yaw_gate = std::stod(argv[9]);
    }

    std::vector<Measurement> measurements;

    try {
        measurements = readCsv(input_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (measurements.empty()) {
        std::cerr << "No measurement data." << std::endl;
        return 1;
    }

    Eigen::VectorXd x0(8);
    x0 <<
        measurements[0].x,
        measurements[0].y,
        measurements[0].z,
        normalizeAngle(measurements[0].yaw),
        0.0,
        0.0,
        0.0,
        0.0;

    Eigen::VectorXd initial_uncertainty(8);
    initial_uncertainty <<
        1e-2, 1e-2, 1e-2, 1e-2,
        1.0, 1.0, 1.0, 1.0;

    Eigen::MatrixXd P0 =
        initial_uncertainty.asDiagonal();

    // yaw 状态更新后重新限制到 [-pi, pi]
    auto x_add = [](
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& delta) {
        Eigen::VectorXd result = state + delta;
        result(3) = normalizeAngle(result(3));
        return result;
    };

    // yaw 残差使用角度差，避免角度回绕造成错误
    auto z_subtract = [](
        const Eigen::VectorXd& measurement,
        const Eigen::VectorXd& predicted) {
        Eigen::VectorXd residual =
            measurement - predicted;

        if (residual.size() >= 4) {
            residual(3) =
                normalizeAngle(residual(3));
        }

        return residual;
    };

    tools::ExtendedKalmanFilter filter(
        x0,
        P0,
        x_add
    );

    Eigen::MatrixXd F =
        makeTransitionMatrix(config.dt);

    Eigen::MatrixXd H =
        makeObservationMatrix();

    Eigen::MatrixXd Q =
        makeProcessNoise(config);

    Eigen::MatrixXd R =
        makeMeasurementNoise(config);

    std::vector<Eigen::VectorXd> filtered_states;
    filtered_states.reserve(measurements.size());

    int yaw_reject_count = 0;

    for (size_t i = 0; i < measurements.size(); ++i) {
        const Measurement& measurement = measurements[i];

        // 预测
        filter.predict(F, Q);
        filter.x(3) = normalizeAngle(filter.x(3));

        Eigen::VectorXd z(4);
        z <<
            measurement.x,
            measurement.y,
            measurement.z,
            measurement.yaw;

        // 计算 yaw 观测与预测之间的角度残差
        Eigen::VectorXd predicted_z = H * filter.x;

        double yaw_residual = normalizeAngle(
            z(3) - predicted_z(3)
        );

        bool yaw_outlier =
            std::abs(yaw_residual) > config.yaw_gate;

        if (yaw_outlier) {
            /*
             * yaw 观测出现接近 180° 的跳变。
             * 这一帧只使用 x、y、z，不使用错误的 yaw。
             */
            Eigen::VectorXd z_position(3);
            z_position <<
                measurement.x,
                measurement.y,
                measurement.z;

            Eigen::MatrixXd H_position =
                Eigen::MatrixXd::Zero(3, 8);
            H_position(0, 0) = 1.0;
            H_position(1, 1) = 1.0;
            H_position(2, 2) = 1.0;

            Eigen::MatrixXd R_position =
                Eigen::MatrixXd::Zero(3, 3);
            R_position(0, 0) = config.r_position;
            R_position(1, 1) = config.r_position;
            R_position(2, 2) = config.r_position;

            filter.update(
                z_position,
                H_position,
                R_position
            );

            ++yaw_reject_count;
        } else {
            filter.update(
                z,
                H,
                R,
                z_subtract
            );
        }

        filter.x(3) = normalizeAngle(filter.x(3));
        filtered_states.push_back(filter.x);
    }

    try {
        writeCsv(output_path, filtered_states);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Measurements: "
              << measurements.size() << std::endl;

    std::cout << "Rejected yaw outliers: "
              << yaw_reject_count << std::endl;

    std::cout << "Filtered data saved to: "
              << output_path << std::endl;

    return 0;
}
