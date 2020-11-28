#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
  Set the process and measurement noises
   */
    H_laser_ << 1, 0, 0, 0,
                  0, 1, 0, 0;

    ekf_.P_ = MatrixXd(4,4);
    ekf_.P_ << 10,  0,   0,   0,
                0,   10,  0,   0,
                0,   0,   100, 0,
                0,   0,   0,   100;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * You'll need to convert radar from polar to cartesian coordinates.
     */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      //         and initialize state.
      float rho     = measurement_pack.raw_measurements_[0]; // range
      float phi     = measurement_pack.raw_measurements_[1]; // bearing
      float rho_dot = measurement_pack.raw_measurements_[2]; // rate

      // Normalize phi to [-pi, pi]
      while (phi > M_PI)  phi -= 2.0 * M_PI;
      while (phi < -M_PI) phi += 2.0 * M_PI;

      // Convert each coordinate
      float x  = rho * cos(phi);
      float y  = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);

      ekf_.x_ << x, y, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      float x = measurement_pack.raw_measurements_[0];
      float y = measurement_pack.raw_measurements_[1];
      float vx = 0;
      float vy = 0;
      ekf_.x_ << x, y, vx, vy;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * Time is measured in seconds.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float dt = (measurement_pack.timestamp_ - previous_timestamp_);
  dt /= 1000000.0; // dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0 , 0, 1, 0,
             0, 0, 0, 1;

   float noise_ax = 9.0;
   float noise_ay = 9.0;

   float dt_2   = dt * dt;
   float dt_3   = dt_2 * dt;
   float dt_4   = dt_3 * dt;
   float dt_4_4 = dt_4 / 4;
   float dt_3_2 = dt_3 / 2;

    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4_4*noise_ax, 0,               dt_3_2*noise_ax, 0,
               0,               dt_4_4*noise_ay, 0,               dt_3_2*noise_ay,
               dt_3_2*noise_ax, 0,               dt_2*noise_ax,   0,
               0,               dt_3_2*noise_ay, 0,               dt_2*noise_ay;

    ekf_.Predict();

  /**
   * Update
   */

  /**
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */


  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.Update(measurement_pack.raw_measurements_);

  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
