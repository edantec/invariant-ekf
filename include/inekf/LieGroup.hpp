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
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>

namespace inekf {

extern const double TOLERANCE;

/**
 * @brief SE3 Lie group with varying Euclidean states.
 *
 */
class SE3_K {
public:
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

  Eigen::Matrix3d skew(Eigen::Ref<const Eigen::Vector3d> v);
  Eigen::Matrix3d exp_SO3(Eigen::Ref<const Eigen::Vector3d> w);
  Eigen::MatrixXd exp_SEK3(Eigen::Ref<const Eigen::VectorXd> v);
  Eigen::MatrixXd adjoint_SEK3(Eigen::Ref<const Eigen::MatrixXd> X);

  void exp_SEK3(Eigen::Ref<const Eigen::VectorXd> v,
                Eigen::Ref<Eigen::MatrixXd> X);
  void adjoint_SEK3(Eigen::Ref<const Eigen::MatrixXd> X,
                    Eigen::Ref<Eigen::MatrixXd> Adj);

protected:
  Eigen::MatrixXd state_;
  Eigen::Matrix3d A_;
  Eigen::Matrix3d A2_;
  Eigen::Matrix3d R_;
  Eigen::Matrix3d Jl_;
  Eigen::Vector3d w_;
};

} // namespace inekf
#endif
