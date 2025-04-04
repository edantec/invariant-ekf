/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   RobotState.hpp
 *  @author Ross Hartley
 *  @brief  Header file for RobotState
 *  @date   September 25, 2018
 **/

#ifndef ROBOTSTATE_H
#define ROBOTSTATE_H
#include <Eigen/Dense>
#include <iostream>
#if INEKF_USE_MUTEX
#include <mutex>
#endif

namespace inekf {

class RobotState {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RobotState();
  RobotState(const Eigen::MatrixXd &X);
  RobotState(const Eigen::MatrixXd &X, const Eigen::VectorXd &Theta);
  RobotState(const Eigen::MatrixXd &X, const Eigen::VectorXd &Theta,
             const Eigen::MatrixXd &P);

#if INEKF_USE_MUTEX
  // RobotState(RobotState&& other); // Move initialization
  RobotState(const RobotState &other); // Copy initialization
  // RobotState& operator=(RobotState&& other); // Move assignment
  RobotState &operator=(const RobotState &other); // Copy assignment
#endif

  const Eigen::MatrixXd getX();
  const Eigen::VectorXd getTheta();
  const Eigen::MatrixXd getP();
  const Eigen::Matrix3d getRotation();
  const Eigen::Vector3d getVelocity();
  const Eigen::Vector3d getPosition();
  const Eigen::Vector3d getGyroscopeBias();
  const Eigen::Vector3d getAccelerometerBias();
  long dimX();
  long dimTheta();
  long dimP();

  void setX(Eigen::Ref<const Eigen::MatrixXd> X);
  void setP(Eigen::Ref<const Eigen::MatrixXd> P);
  void setTheta(Eigen::Ref<const Eigen::VectorXd> Theta);
  void setRotation(Eigen::Ref<const Eigen::Matrix3d> R);
  void setVelocity(Eigen::Ref<const Eigen::Vector3d> v);
  void setPosition(Eigen::Ref<const Eigen::Vector3d> p);
  void setGyroscopeBias(Eigen::Ref<const Eigen::Vector3d> bg);
  void setAccelerometerBias(Eigen::Ref<const Eigen::Vector3d> ba);

  void copyDiagX(int n, Eigen::MatrixXd &BigX);

  friend std::ostream &operator<<(std::ostream &os, const RobotState &s);

private:
  Eigen::MatrixXd X_;
  Eigen::VectorXd Theta_;
  Eigen::MatrixXd P_;
#if INEKF_USE_MUTEX
  mutable std::mutex mutex_;
#endif
};

} // namespace inekf
#endif
