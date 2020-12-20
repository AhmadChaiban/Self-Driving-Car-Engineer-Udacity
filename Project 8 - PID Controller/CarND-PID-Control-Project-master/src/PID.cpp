#include "PID.h"
#include <vector>
#include <numeric>
#include <string>
#include "json.hpp"
#include <uWS/uWS.h>
using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
    /**
    * Initialize PID coefficients (and errors, if needed)
    */
    Kp = Kp_;
    Ki = Ki_;
    Kd = Kd_;
}

double PID::CalculateSteer(){
    double current_cte = cte_vector[cte_vector.size() - 1];
    double steer_value = -Kp * current_cte -Kd * (current_cte - prev_cte) -Ki * int_cte;
    prev_cte = current_cte;
    return steer_value;
}

void PID::UpdateError(double cte) {
    /**
    * Update PID errors based on cte.
    */
    cte_vector.push_back(cte);
}

double PID::TotalError() {
    /**
    * Calculate and return the total error
    */

    // double int_cte = 0.0;
    // for(int i=0; i<cte_vector.size(); i++){
    //     int_cte += cte_vector[i];
    // }
    // return int_cte;  // Add total error
    int_cte += cte_vector[cte_vector.size() - 1];
    err_sum += (cte_vector[cte_vector.size() - 1] * cte_vector[cte_vector.size() - 1])/(cte_vector.size());

}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Running Twiddle here, however, it's still in progress at this stage. I manually tuned
// the parameters for now.

// I'm attempting to send commands to the websocket at this stage.

/*
 ____________________
/                    \
|     Warning!       |
|  Unfinished code   |
|      below!         |
\____________________/
         !  !
         !  !
         L_ !
        / _)!
       / /__L
 _____/ (____)
        (____)
 _____  (____)
      \_(____)
         !  !
         !  !
         \__/
 */

//////////////////////////////////////////////////////////////////////////////////////////////////

void PID::Twiddle(double tolerance, double cte, uWS::WebSocket<uWS::SERVER> ws, std::string s){
    //Parameter optimization
    double best_err;
    std::vector<double> p = {0.0, 0.0, 0.0};
    std::vector<double> dp = {1.0, 1.0, 1.0};
    Run(w, s)
    best_err = err_sum;
    while(accumulate(dp.begin(), dp.end(), 0.0) > tolerance){
        for(int i=0; i<p.size(); i++){
            p[i] += dp[i];
            if(err_sum < best_err){
                best_err = err_sum;
                dp[i] *= 1.1;
            }
            else{
                p[i] -= 2.0 * dp[i];
                if(err_sum < best_err){
                    best_err = err_sum;
                    dp[i] *= 1.1;
                }
                else{
                    p[i] += dp[i];
                    dp[i] *= 0.9;
                }
            }

        }
    }
    Kp = p[0]; Kd = p[1]; Ki = p[2];
    std::cout << "Kp = " << Kp << ", Kd = " << Kd << ", Ki = " << Ki << std::endl;
}

void PID::ResetCar(uWS::WebSocket<uWS::SERVER> ws){
    nlohmann::json msgJson;
    auto msg = "42[\"reset\",{}]";
    ws.send(msg, uWS::OpCode::TEXT);
}

void PID::MoveCar(uWS::WebSocket<uWS::SERVER> ws, std::string s){
    nlohmann::json msgJson;
    double steer_value = CalculateSteer();
    msgJson["steering_angle"] = steer_value;
    msgJson["throttle"] = 0.3;
    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

    auto j = nlohmann::json::parse(s);
    double cte = std::stod(j[1]["cte"].get<string>());
    UpdateError(cte);
    TotalError();
}

void PID::Run(uWS::WebSocket<uWS::SERVER> ws, std::string s){
    for(int j=0; j<100000000; j++){
        MoveCar(ws, s);
    }
    ResetCar(ws);
}