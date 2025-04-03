/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   LieGroup.hpp
 *  @author Ross Hartley
 *  @brief  Header file for various Lie Group functions
 *  @date   September 25, 2018
 **/

#ifndef LIEGROUP_H
#define LIEGROUP_H
#include <Eigen/Dense>

namespace inekf {

extern const double TOLERANCE;

/**
 * @brief SE3 Lie group with varying Euclidean states.
 *
 * @tparam K maximum number of augmented Euclidean states.
 */
template <int K> class SE3_K {
public:
  /**
   * @brief Matrix state includes base rotation, velocity and position
   * as well as Euclidean states
   *
   */
  typedef Eigen::Matrix<double, 5 + K, 5 + K> MatrixState;
  /**
   * @brief Related tangent vector
   *
   */
  typedef Eigen::Matrix<double, 9 + K, 1> TangentVector;

  SE3_K() {}

  /**
   * @brief Class constructor

   * @param rotation Base rotation
   * @param position Base position
   *
   */
  SE3_K(Eigen::Matrix3d &rotation, Eigen::Vector3d &position);

  /**
   * @brief Class constructor, initialized with null velocity

   * @param rotation Base rotation
   * @param position Base position
   * @param feet_positions Vector of 3d poses of max size N
   *
   */
  SE3_K(Eigen::Matrix3d &rotation, Eigen::Vector3d &position,
        std::vector<Eigen::Vector3d> feet_translation);

  virtual ~SE3_K() {};

protected:
  MatrixState state_;
};

Eigen::Matrix3d skew(const Eigen::Vector3d &v);
Eigen::Matrix3d exp_SO3(const Eigen::Vector3d &w);
Eigen::MatrixXd exp_SEK3(const Eigen::VectorXd &v);
Eigen::MatrixXd adjoint_SEK3(const Eigen::MatrixXd &X);

void exp_SEK3(const Eigen::VectorXd &v, Eigen::Ref<Eigen::MatrixXd> X);
void adjoint_SEK3(const Eigen::MatrixXd &X, Eigen::Ref<Eigen::MatrixXd> Adj);

} // namespace inekf
#endif
