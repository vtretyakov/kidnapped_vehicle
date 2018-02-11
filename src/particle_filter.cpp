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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    //Set the number of particles
    num_particles = 10; //start with some not too high number and later tune
    
    default_random_engine gen;

    //Create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);//std_x
    normal_distribution<double> dist_y(y, std[1]);//std_y
    normal_distribution<double> dist_theta(theta, std[2]);//std_theta
    
    for (int i = 0; i < num_particles; i++){
        
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;
        
        particles.push_back(particle);

    }
    
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    
    // Distributions for adding random Gaussian noise
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    for (int i = 0; i < num_particles; i++){
    
        //Update x, y and the yaw angle
        if (fabs(yaw_rate) < 0.0001) {//avoid devision by zero
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
            //do not update theta due to no yaw rate
        } else {
            double new_theta = particles[i].theta + yaw_rate * delta_t;
            particles[i].x += (velocity / yaw_rate) * (sin(new_theta) - sin(particles[i].theta));
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(new_theta));
            particles[i].theta = new_theta;
        }
        
        //Add Gaussian noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    // For each observation, search the closest predicted distance
    for (unsigned int i = 0; i < observations.size(); i++){
    
        double min_dist = 100000; // start with an unrealisticly high distance
        for (unsigned int j = 0; j < predicted.size(); j++){
        
            double d = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
            // if it is closest, update minimum distance and id.
            if (d < min_dist)
            {
                min_dist = d;
                observations[i].id = predicted[j].id;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
    
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    
    // Accumulated weights for normalization
    double weight_acc = 0;
    
    // Clear ParticleFilter weights
    weights.clear();
    
    for (int i = 0; i < num_particles; i++) {
        
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        // Find landmarks in particle's range
        vector<LandmarkObs> predictions;
        for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            double l_x = map_landmarks.landmark_list[j].x_f;
            double l_y = map_landmarks.landmark_list[j].y_f;
            int l_id = map_landmarks.landmark_list[j].id_i;
            double distance = dist(l_x, l_y, p_x, p_y);
            if ( distance <= sensor_range ) {
                predictions.push_back(LandmarkObs{ l_id, l_x, l_y });
            }
        }
        
        // Convert observations from vehicle to map coordinates
        vector<LandmarkObs> global_observations;
        for(unsigned int j = 0; j < observations.size(); j++) {
            double transformed_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
            double transformed_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
            global_observations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
        }
        
        // Observation association to landmark
        dataAssociation(predictions, global_observations);
        
        // Initialize probability
        double prob = 1;
        
        // Calculate the posterior probability for each observation
        for (unsigned int j = 0; j < global_observations.size(); j++){
            // Search for predictions associated with this landamark id:
            int id = global_observations[j].id;
            double obs_x = global_observations[j].x;
            double obs_y = global_observations[j].y;
            
            for (unsigned int n = 0; n < predictions.size(); n++){
                if(predictions[n].id == id){
                    prob *= bivariateGaussianPdf(obs_x, obs_y, predictions[n].x, predictions[n].y, std_x, std_y, 0);
                    break;//stop further searching, we've found
                }
            }
        }
        particles[i].weight = prob;
        
        // Sum up weights
        weight_acc += prob;
    }
    
    // Normalize weights
    for (int i = 0; i < num_particles; i++)
    {
        particles[i].weight /= weight_acc;
        weights.push_back(particles[i].weight);
    }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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

double ParticleFilter::bivariateGaussianPdf(double x, double y, double mean_x, double mean_y, double std_x, double std_y, double rho){
    
    if (fabs(rho) >= 1) rho = 0;//avoid devision by zero with invalid input (-1 < rho < 1)
    double std_x_2 = std_x*std_x;
    double std_y_2 = std_y*std_y;
    double std_xy = std_x*std_y;
    double rho_2 = rho*rho;
    double denom = 2*M_PI*std_xy*sqrt(1-rho_2);
    double dx = x-mean_x;
    double dy = y-mean_y;
    double exponent = ( dx*dx/std_x_2 + dy*dy/std_y_2 - 2*rho*dx*dy/std_xy )/(2*(1-rho_2));
    double prob = exp(-exponent)/denom;
    
    return prob;
}
