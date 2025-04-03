/* ----------------------------------------------------------------------------
 * Copyright 2025, Ewen Dantec
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   lie_group.cpp
 *  @author Ewen Dantec
 *  @brief  Test Lie group functions
 *  @date   2025
 **/

#include <boost/test/unit_test.hpp>

#include "inekf/LieGroup.hpp"
#include "inekf/utils.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#define DT_MIN 1e-6
#define DT_MAX 1
#define TOLERANCE 1e-6

BOOST_AUTO_TEST_SUITE(lie_group)

using namespace std;
using namespace inekf;

BOOST_AUTO_TEST_CASE(SO3) {
  // Check SO3
  Eigen::Vector3d w;
  w.setRandom();

  Eigen::Matrix3d X = exp_SO3(w);
  Eigen::Matrix3d ref = skew(w).exp();

  BOOST_CHECK(ref.isApprox(X, TOLERANCE));
}

BOOST_AUTO_TEST_CASE(Adj) {
  // Check adjoint
  Eigen::VectorXd w(15);
  w.setRandom();
  Eigen::MatrixXd X = exp_SEK3(w);
  Eigen::MatrixXd Adj = adjoint_SEK3(X);

  Eigen::Matrix3d top;

  for (int i = 0; i < 5; i++) {
    BOOST_CHECK(
        Adj.block(3 * i, 3 * i, 3, 3).isApprox(X.block(0, 0, 3, 3), TOLERANCE));
  }
  for (int i = 0; i < 4; i++) {
    top = skew(X.block(0, 3 + i, 3, 1)) * X.block(0, 0, 3, 3);
    BOOST_CHECK(Adj.block(3 + 3 * i, 0, 3, 3).isApprox(top, TOLERANCE));
  }

  long K = X.cols() - 3;
  Eigen::MatrixXd Adj2(3 + 3 * K, 3 + 3 * K);
  Adj2.setIdentity();
  adjoint_SEK3(X, Adj2);
  BOOST_CHECK(Adj.isApprox(Adj2, TOLERANCE));
}

BOOST_AUTO_TEST_CASE(exp) {
  // Check exp_SE3
  Eigen::VectorXd w(15);
  w.setRandom();
  Eigen::MatrixXd X = exp_SEK3(w);

  long N = (w.size() - 3) / 3;

  Eigen::MatrixXd Skew = Eigen::MatrixXd::Zero(3 + N, 3 + N);
  Skew.topLeftCorner(3, 3) = skew(w.head(3));
  for (int i = 0; i < 4; i++) {
    Skew.block(0, 3 + i, 3, 1) = w.segment(3 * i + 3, 3);
  }
  Eigen::MatrixXd Expref = Skew.exp();

  BOOST_CHECK(Expref.isApprox(X, TOLERANCE));

  Eigen::MatrixXd Xn(3 + N, 3 + N);
  exp_SEK3(w, Xn);

  BOOST_CHECK(Xn.isApprox(X, TOLERANCE));
}

BOOST_AUTO_TEST_SUITE_END()
