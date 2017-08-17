/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  //TODO: Choose a better M.
  num_particles = 1000;
  default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  double sample_x, sample_y, sample_theta;
  particles.clear();
  weights.clear();
  
  for (int i = 0; i < num_particles; ++i) {
    //  Sample x,y , theta from these normal distrubtions
    //  where "gen" is the random engine initialized earlier.
     sample_x = dist_x(gen);
     sample_y = dist_y(gen);
     sample_theta = dist_theta(gen);   
     
     // Print your samples to the terminal.
     cout << "Sample " << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << endl;

     const double default_weight = 1.0;
     //Append particles in class attribute
     particles.emplace_back(Particle{i, sample_x, sample_y, sample_theta, default_weight});
     weights.emplace_back(default_weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  double pred_x, pred_y, pred_theta;
  
  //Calculate predictions
  for(int i = 0; i < num_particles; ++i){
    if(yaw_rate == 0.0){
      double distance = velocity*delta_t;
      pred_theta = particles[i].theta;
      pred_x = particles[i].x + distance*cos(pred_theta);
      pred_y = particles[i].y + distance*sin(pred_theta);
    }
    else{
      double t_velocity = velocity/yaw_rate;
      pred_theta = particles[i].theta + delta_t*yaw_rate;
      pred_x = particles[i].x + t_velocity*(sin(pred_theta) - sin(particles[i].theta));
      pred_y = particles[i].y + t_velocity*(cos(particles[i].theta) - cos(pred_theta));
    }
    normal_distribution<double> dist_x(pred_x, std_pos[0]);
    normal_distribution<double> dist_y(pred_y, std_pos[1]);
    normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

double ParticleFilter::dataAssociation(std::vector<LandmarkObs> map_landmarks, std::vector<LandmarkObs>& observations, double std_landmark[]) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // std::cout << "# observations: " << observations.size() << std::endl;
  // std::cout << "# landmarks in range: " << observations.size() << std::endl;
  double weight = 1.0;
  double prob_norm = 2*M_PI*std_landmark[0]*std_landmark[1];
  double distance_to_landmark;
  double closest_dist, dx_2, dy_2;
  double exp_x, exp_y, prob;
  for(auto &obs : observations){
    //Initialized best distance with dist to first landmark
    closest_dist = dist(map_landmarks[0].x, map_landmarks[0].y, obs.x, obs.y);
    dx_2 = map_landmarks[0].x - obs.x;
    dy_2 = map_landmarks[0].y - obs.y;

    // Search for the closest landmark
    for(auto &map_landmark : map_landmarks){
      distance_to_landmark = dist( map_landmark.x, map_landmark.y, obs.x, obs.y);
      if(distance_to_landmark < closest_dist){
        closest_dist = distance_to_landmark;
        obs.id = map_landmark.id;
        dx_2 = map_landmark.x - obs.x;
        dy_2 = map_landmark.y - obs.y;
      }
    }
    dx_2 *= dx_2;
    dy_2 *= dy_2;

    exp_x = dx_2 / (2*std_landmark[0]*std_landmark[0]);
    exp_y = dy_2 / (2*std_landmark[1]*std_landmark[1]);
    prob = exp(-(exp_x + exp_y)) / prob_norm;
    weight *= prob;
    // std::cout << "closest landmark: " << obs.id << "\tweight: " << weight << std::endl;
    // std::cout << "prob: " << prob_norm << "\tw: " << weight << std::endl;
  }
  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  std::vector<LandmarkObs> map_observations;
  for (int p = 0; p < num_particles; ++p){
    map_observations.clear();
    double map_x;     // Local (vehicle coordinates) x position of landmark observation [m]
    double map_y;     // Local (vehicle coordinates) y position of landmark observation [m]
    
    for ( auto &obs : observations) {
      //Rotate observation using particle.theta and translate using particle's map coordinate
      map_x = obs.x * cos(particles[p].theta) - obs.y * sin(particles[p].theta) + particles[p].x;
      map_y = obs.x * sin(particles[p].theta) + obs.y * cos(particles[p].theta) + particles[p].y;
      map_observations.push_back(LandmarkObs{obs.id, map_x, map_y});
    }

    std::vector<LandmarkObs> landmarks_in_range;
    for(auto &map_landmark : map_landmarks.landmark_list){
      double distance_to_particle = dist( map_landmark.x_f, map_landmark.y_f, particles[p].x, particles[p].y);
      if(distance_to_particle < sensor_range){
        landmarks_in_range.emplace_back(LandmarkObs{map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
      }
    }

    particles[p].weight = dataAssociation(landmarks_in_range , map_observations, std_landmark);

    // std::vector<int> associations;
    // std::vector<double> sense_x;
    // std::vector<double> sense_y;
    // for ( auto &obs : map_observations) {
    //   associations.push_back(obs.id);
    //   sense_x.push_back(obs.x);
    //   sense_y.push_back(obs.y);
    // }

    //particles[p] = SetAssociations(particles[p], associations, sense_x, sense_y);
    weights[p] = particles[p].weight;

  }


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;

  // This line creates a normal (Gaussian) distribution for x, y and theta
  discrete_distribution<int> distribution(weights.begin(), weights.end());
  vector<Particle> resampled_particles;

   
  for (int i = 0; i < num_particles; ++i) {
     // Print your samples to the terminal.
     resampled_particles.push_back(particles[distribution(gen)]);
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
