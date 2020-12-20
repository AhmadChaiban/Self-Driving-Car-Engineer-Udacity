#ifndef PID_H
#define PID_H

#include <vector>
#include <uWS/uWS.h>
#include <string>

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the steering angle using the private variables
   */
  double CalculateSteer();

  /**
   * vector to keep track of ctes
   */
  std::vector<double> cte_vector;

  /**
   * sum of previous CTEs
   */
  double int_cte = 0.0;
  double err_sum = 0.0;

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  /**
   *
   * @param ws
   * Sending commands to the vehicles
   */
  void MoveCar(uWS::WebSocket<uWS::SERVER> ws, std::string s);
  void ResetCar(uWS::WebSocket<uWS::SERVER> ws);

  /**
   * Run some iterations on the moving car
   * @param ws
   * @param s
   */

    void Run(uWS::WebSocket<uWS::SERVER> ws, std::string s);
  /*
   * Twiddle function for parameter optimization
   */

  void Twiddle(double tolerance, double cte, uWS::WebSocket<uWS::SERVER> ws, std::string s);

 private:
  /**
   * PID Errors
   */
  double prev_cte = 0.0;

  /**
   * PID Coefficients
   */ 
  double Kp;
  double Ki;
  double Kd;
};

#endif  // PID_H