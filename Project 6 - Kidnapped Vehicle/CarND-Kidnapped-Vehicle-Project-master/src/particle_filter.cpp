/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    num_particles = 100;
    particles.reserve(num_particles);

    normal_distribution<double> dist_x(0, std[0]);
    normal_distribution<double> dist_y(0, std[1]);
    normal_distribution<double> dist_theta(0, std[2]);

    for(int i=0; i<num_particles; i++){
        particles[i].id = i;
        particles[i].x = x + dist_x(gen);
        particles[i].y = y + dist_y(gen);
        particles[i].theta = theta + dist_theta(gen);
        particles[i].weight = 1.0;


    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

    for(int i=0; i<num_particles; i++){

        normal_distribution<double> dist_x_p(0, std_pos[0]);
        normal_distribution<double> dist_y_p(0, std_pos[1]);
        normal_distribution<double> dist_theta_p(0, std_pos[2]);

        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else{
            double new_theta = particles[i].theta + yaw_rate * delta_t;
            particles[i].x += (velocity/yaw_rate) * (sin(new_theta) - sin(particles[i].theta)) + dist_x_p(gen);
            particles[i].y += (velocity/yaw_rate) * (-cos(new_theta) + cos(particles[i].theta)) + dist_y_p(gen);
            particles[i].theta = new_theta + dist_theta_p(gen);
        }
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

    for (int i = 0; i < observations.size(); i++) {
        double min_dist = std::numeric_limits<double>::max();
        int map_id = -1;

        for (unsigned int j = 0; j < predicted.size(); j++) {
            double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                map_id = predicted[j].id;
            }
        }
        observations[i].id = map_id;
    }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

    for(int i=0; i<particles.size(); i++) {
        vector <LandmarkObs> predictions;

        vector<LandmarkObs> nearby_landmarks;
        LandmarkObs nearby_landmark;

        vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;

        for(unsigned int k=0; k<landmarks.size(); ++k){
            double landmark_dist = dist(particles[i].x, particles[i].y, landmarks[k].x_f, landmarks[k].y_f);

            if(landmark_dist < sensor_range){
                nearby_landmark.id = landmarks[k].id_i;
                nearby_landmark.x = landmarks[k].x_f;
                nearby_landmark.y = landmarks[k].y_f;
                nearby_landmarks.push_back(nearby_landmark);
            }
        }

        for (int j = 0; j < nearby_landmarks.size(); j++) {
            float l_x = nearby_landmarks[j].x;
            float l_y = nearby_landmarks[j].y;
            int l_id = nearby_landmarks[j].id;

            if (fabs(l_x - particles[i].x) <= sensor_range && fabs(l_y - particles[i].y) <= sensor_range) {

                predictions.push_back(LandmarkObs{ l_id, l_x, l_y});
            }

            vector<LandmarkObs> trans_obs;
            for (int j = 0; j < observations.size(); j++) {
                double t_x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
                double t_y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
                trans_obs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
            }

            dataAssociation(predictions, trans_obs);

            particles[i].weight = 1.0;

            for (int j = 0; j < trans_obs.size(); j++) {

                // placeholders for observation and associated prediction coordinates
                double obs_x, obs_y, pr_x, pr_y;
                obs_x = trans_obs[j].x;
                obs_y = trans_obs[j].y;

                int associated_prediction = trans_obs[j].id;

                // get the x,y coordinates of the prediction associated with the current observation
                for (unsigned int k = 0; k < predictions.size(); k++) {
                    if (predictions[k].id == associated_prediction) {
                        pr_x = predictions[k].x;
                        pr_y = predictions[k].y;
                    }
                }

                double sig_x = std_landmark[0];
                double sig_y = std_landmark[1];
                double obs_w = 1/(2*M_PI*sig_x*sig_y);
                obs_w += exp(-( pow(pr_x-obs_x,2)/(2*pow(sig_x, 2))
                        + (pow(pr_y-obs_y,2)/(2*pow(sig_y, 2)))));

                particles[i].weight *= obs_w;
            }
        }

    }
}

void ParticleFilter::resample() {
  /**
   *   Resample particles with replacement with probability proportional
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    vector<Particle> new_particles;
    vector<double> weights;

    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    std::uniform_int_distribution<int> uniform_int_dist(0, num_particles-1);
    auto index = uniform_int_dist(gen);

    double max_weight = *max_element(weights.begin(), weights.end());

    std::uniform_real_distribution<double> uniform_real_dist(0.0, max_weight);

    double beta = 0.0;

    for (int i = 0; i < num_particles; i++) {
        beta += uniform_real_dist(gen) * 2.0;
        while (beta > weights[index]) {
            beta = beta - weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}