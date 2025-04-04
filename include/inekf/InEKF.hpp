/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   InEKF.hpp
 *  @author Ross Hartley
 *  @brief  Header file for Invariant EKF
 *  @date   September 25, 2018
 **/

#ifndef INEKF_H
#define INEKF_H
#define EIGEN_RUNTIME_NO_MALLOC

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#if INEKF_USE_MUTEX
#include <mutex>
#endif
#include "inekf/LieGroup.hpp"
#include "inekf/NoiseParams.hpp"
#include "inekf/RobotState.hpp"

namespace inekf {

class Kinematics {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Kinematics() {}
  Kinematics(int id_in, const Eigen::MatrixXd &pose_in,
             const Eigen::MatrixXd &covariance_in)
      : id(id_in), pose(pose_in), covariance(covariance_in) {}

  int id;
  Eigen::MatrixXd pose;
  Eigen::MatrixXd covariance;
};

class Landmark {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Landmark(int id_in, const Eigen::Vector3d &position_in)
      : id(id_in), position(position_in) {}

  int id;
  Eigen::Vector3d position;
};

typedef std::map<
    int, Eigen::Vector3d, std::less<int>,
    Eigen::aligned_allocator<std::pair<const int, Eigen::Vector3d>>>
    mapIntVector3d;
typedef std::map<
    int, Eigen::Vector3d, std::less<int>,
    Eigen::aligned_allocator<std::pair<const int, Eigen::Vector3d>>>::iterator
    mapIntVector3dIterator;
typedef std::vector<Landmark, Eigen::aligned_allocator<Landmark>>
    vectorLandmarks;
typedef std::vector<Landmark,
                    Eigen::aligned_allocator<Landmark>>::const_iterator
    vectorLandmarksIterator;
typedef std::vector<Kinematics, Eigen::aligned_allocator<Kinematics>>
    vectorKinematics;
typedef std::vector<Kinematics,
                    Eigen::aligned_allocator<Kinematics>>::const_iterator
    vectorKinematicsIterator;

class Observation {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Observation(const Eigen::VectorXd &Y, const Eigen::VectorXd &b,
              const Eigen::MatrixXd &H, const Eigen::MatrixXd &N,
              const Eigen::MatrixXd &PI)
      : Y(Y), b(b), H(H), N(N), PI(PI) {}
  bool empty() { return Y.rows() == 0; }

  Eigen::VectorXd Y;
  Eigen::VectorXd b;
  Eigen::MatrixXd H;
  Eigen::MatrixXd N;
  Eigen::MatrixXd PI;

  friend std::ostream &operator<<(std::ostream &os, const Observation &o) {
    os << "---------- Observation ------------" << std::endl;
    os << "Y:\n" << o.Y << std::endl << std::endl;
    os << "b:\n" << o.b << std::endl << std::endl;
    os << "H:\n" << o.H << std::endl << std::endl;
    os << "N:\n" << o.N << std::endl << std::endl;
    os << "PI:\n" << o.PI << std::endl;
    os << "-----------------------------------";
    return os;
  }
};

class InEKF {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Constructors
  InEKF() {}
  InEKF(const long contact_nb) : contact_nb_(contact_nb) { setMatrices(); }
  InEKF(const RobotState &state) : state_(state) { setMatrices(); }
  InEKF(const RobotState &state, const NoiseParams &params)
      : state_(state), noise_params_(params) {
    setMatrices();
  }
  InEKF(const RobotState &state, const NoiseParams &params,
        const long contact_nb)
      : state_(state), noise_params_(params), contact_nb_(contact_nb) {
    setMatrices();
  }

  RobotState getState();
  const NoiseParams getNoiseParams();
  const mapIntVector3d getPriorLandmarks();
  const std::map<int, int> getEstimatedLandmarks();
  const std::map<int, bool> getContacts();
  const std::map<int, int> getEstimatedContactPositions();

  void setState(const RobotState &state);
  void setNoiseParams(const NoiseParams &params);
  void setPriorLandmarks(const mapIntVector3d &prior_landmarks);
  void setContacts(std::vector<std::pair<int, bool>> contacts);
  void setGravity(const Eigen::Vector3d &gravity);

  void propagate(const Eigen::VectorXd &m, double dt);
  void correct(const Observation &obs);
  void correctLandmarks(const vectorLandmarks &measured_landmarks);
  void correctKinematics(const vectorKinematics &measured_kinematics);
  void removeRowAndColumn(Eigen::MatrixXd &M, int index);

private:
  RobotState state_;
  NoiseParams noise_params_;
  long contact_nb_ = 0;
  Eigen::MatrixXd dX_;
  Eigen::MatrixXd X_new_;
  Eigen::VectorXd theta_new_;
  Eigen::MatrixXd Adj_;
  Eigen::MatrixXd PHT_;
  Eigen::MatrixXd S_;
  Eigen::MatrixXd K_;
  Eigen::MatrixXd IKH_;
  Eigen::MatrixXd P_inter_;
  Eigen::MatrixXd P_new_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd A2_;
  Eigen::MatrixXd A3_;
  Eigen::MatrixXd Phi_;
  Eigen::VectorXd delta_;
  Eigen::VectorXd z_;
  Eigen::VectorXd PIz_;

  inekf::SE3_K SE3_;
  Eigen::Matrix3d Skew_g_;
  Eigen::Vector3d g_ = Eigen::Vector3d(0, 0, -9.81); // Gravity
  mapIntVector3d prior_landmarks_;
  std::map<int, int> estimated_landmarks_;
  std::map<int, bool> contacts_;
  std::map<int, int> estimated_contact_positions_;

  void setMatrices() {
    dX_.resize(5 + contact_nb_, 5 + contact_nb_);
    dX_.setZero();

    X_new_.resize(5 + contact_nb_, 5 + contact_nb_);
    X_new_.setZero();

    theta_new_.resize(6);
    theta_new_.setZero();

    Adj_.resize((9 + contact_nb_) * 3, (9 + contact_nb_) * 3);
    Adj_.setZero();

    PHT_.resize((5 + contact_nb_) * 3, 3 * contact_nb_);
    PHT_.setZero();

    S_.resize(3 * contact_nb_, 3 * contact_nb_);
    S_.setZero();

    K_.resize((5 + contact_nb_) * 3, 3 * contact_nb_);
    K_.setZero();

    IKH_.resize((5 + contact_nb_) * 3, (5 + contact_nb_) * 3);
    IKH_.setZero();

    P_inter_.resize((5 + contact_nb_) * 3, (5 + contact_nb_) * 3);
    P_inter_.setZero();

    P_new_.resize((5 + contact_nb_) * 3, (5 + contact_nb_) * 3);
    P_new_.setZero();

    A_.resize(state_.dimP() + contact_nb_ * 3, state_.dimP() + contact_nb_ * 3);
    A_.setZero();

    A2_.resize(state_.dimP() + contact_nb_ * 3,
               state_.dimP() + contact_nb_ * 3);
    A2_.setZero();

    A3_.resize(state_.dimP() + contact_nb_ * 3,
               state_.dimP() + contact_nb_ * 3);
    A3_.setZero();

    Phi_.resize(state_.dimP() + contact_nb_ * 3,
                state_.dimP() + contact_nb_ * 3);
    Phi_.setZero();

    delta_.resize((5 + contact_nb_) * 3);
    delta_.setZero();

    z_.resize(7 * contact_nb_);
    z_.setZero();

    PIz_.resize(3 * contact_nb_);
    PIz_.setZero();

    Skew_g_ = SE3_.skew(g_);
  };
#if INEKF_USE_MUTEX
  std::mutex estimated_contacts_mutex_;
  std::mutex estimated_landmarks_mutex_;
#endif
};

} // namespace inekf
#endif
