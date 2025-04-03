/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley <m.ross.hartley@gmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   LieGroup.cpp
 *  @author Ross Hartley
 *  @brief  Source file for various Lie Group functions
 *  @date   September 25, 2018
 **/

#include "inekf/LieGroup.hpp"

namespace inekf {

using namespace std;

const double TOLERANCE = 1e-10;

template <int K>
SE3_K<K>::SE3_K(Eigen::Matrix3d &rotation, Eigen::Vector3d &position) {
  state_.setIdentity();
  state_.block(0, 0, 3, 3) = rotation;
  state_.block(0, 3, 3, 1) = position;
}

template <int K>
SE3_K<K>::SE3_K(Eigen::Matrix3d &rotation, Eigen::Vector3d &position,
                vector<Eigen::Vector3d> feet_translation) {

  assert((feet_translation.size() <= K));

  state_.setIdentity();
  state_.block(0, 0, 3, 3) = rotation;
  state_.block(0, 3, 3, 1) = position;

  for (unsigned int i = 0; i < feet_translation.size(); i++) {
    state_.block(0, 5 + i, 3, 1) = feet_translation[i];
  }
}

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  // Convert vector to skew-symmetric matrix
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  M << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return M;
}

Eigen::Matrix3d exp_SO3(const Eigen::Vector3d &w) {
  // Computes the vectorized exponential map for SO(3)
  Eigen::Matrix3d A = skew(w);
  Eigen::Matrix3d A2 = A * A;
  double theta = w.norm();
  if (theta < TOLERANCE) {
    return Eigen::Matrix3d::Identity() + A / 2 + A2 / 6 + A2 * A / 24;
  }
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() + (sin(theta) / theta) * A +
                      ((1 - cos(theta)) / (theta * theta)) * A2;
  return R;
}

Eigen::MatrixXd exp_SEK3(const Eigen::VectorXd &v) {
  // Computes the vectorized exponential map for SE_K(3)
  long N = (v.size() - 3) / 3;

  Eigen::MatrixXd X = Eigen::MatrixXd::Identity(3 + N, 3 + N);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d Jl = Eigen::Matrix3d::Identity();
  Eigen::Vector3d w = v.head(3);
  double theta = w.norm();
  if (theta >= TOLERANCE) {
    Eigen::Matrix3d A = skew(w);
    Eigen::Matrix3d A2 = A * A;
    double theta2 = theta * theta;
    double stheta = sin(theta);
    double ctheta = cos(theta);
    double oneMinusCosTheta2 = (1 - ctheta) / (theta2);
    R += (stheta / theta) * A + oneMinusCosTheta2 * A2;
    Jl += oneMinusCosTheta2 * A + ((theta - stheta) / (theta2 * theta)) * A2;
  }
  X.block<3, 3>(0, 0) = R;
  for (int i = 0; i < N; ++i) {
    X.block<3, 1>(0, 3 + i) = Jl * v.segment<3>(3 + 3 * i);
  }

  return X;
}

void exp_SEK3(const Eigen::VectorXd &v, Eigen::Ref<Eigen::MatrixXd> X) {
  // Computes the vectorized exponential map for SE_K(3)
  long N = (v.size() - 3) / 3;

  assert((X.rows() == 3 + N && X.cols() == 3 + N));
  X.setIdentity();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d Jl = Eigen::Matrix3d::Identity();
  Eigen::Vector3d w = v.head(3);
  double theta = w.norm();
  if (theta >= TOLERANCE) {
    Eigen::Matrix3d A = skew(w);
    Eigen::Matrix3d A2 = A * A;
    double theta2 = theta * theta;
    double stheta = sin(theta);
    double ctheta = cos(theta);
    double oneMinusCosTheta2 = (1 - ctheta) / (theta2);
    R += (stheta / theta) * A + oneMinusCosTheta2 * A2;
    Jl += oneMinusCosTheta2 * A + ((theta - stheta) / (theta2 * theta)) * A2;
  }
  X.block<3, 3>(0, 0) = R;
  for (int i = 0; i < N; ++i) {
    X.block<3, 1>(0, 3 + i) = Jl * v.segment<3>(3 + 3 * i);
  }
}

Eigen::MatrixXd adjoint_SEK3(const Eigen::MatrixXd &X) {
  // Compute Adjoint(X) for X in SE_K(3)
  long K = X.cols() - 3;
  Eigen::MatrixXd Adj = Eigen::MatrixXd::Identity(3 + 3 * K, 3 + 3 * K);
  const Eigen::Matrix3d &R = X.block<3, 3>(0, 0);
  Adj.block<3, 3>(0, 0) = R;
  for (int i = 0; i < K; ++i) {
    Adj.block<3, 3>(3 + 3 * i, 3 + 3 * i) = R;
    Adj.block<3, 3>(3 + 3 * i, 0) = skew(X.block<3, 1>(0, 3 + i)) * R;
  }
  return Adj;
}

void adjoint_SEK3(const Eigen::MatrixXd &X, Eigen::Ref<Eigen::MatrixXd> Adj) {
  // Compute Adjoint(X) for X in SE_K(3)
  long K = X.cols() - 3;
  assert((Adj.rows() == 3 + 3 * K && Adj.cols() == 3 + 3 * K));
  Adj.setZero();
  const Eigen::Matrix3d &R = X.block<3, 3>(0, 0);
  Adj.block<3, 3>(0, 0) = R;
  for (int i = 0; i < K; ++i) {
    Adj.block<3, 3>(3 + 3 * i, 3 + 3 * i) = R;
    Adj.block<3, 3>(3 + 3 * i, 0) = skew(X.block<3, 1>(0, 3 + i)) * R;
  }
}

} // namespace inekf
