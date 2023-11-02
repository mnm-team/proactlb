
#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>

#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>

#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>

#include <ensmallen.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace mlpack;
using namespace ens;

/**
 * Read csv file and parse to a matrix format
 *
 * Can be re-format with other type of data
 *
 * @param &filename: path to the input
 * @param &delimeter: default is ","
 * @return matrix format by the class from arma::mat
 */
arma::mat read_mat_csv(const std::string &filename, const std::string &delimeter = ","){

    // file name & data
    std::ifstream csv_file(filename);
    std::vector<std::vector<double>> data_batch;

    // read file line by line
    for(std::string line; std::getline(csv_file, line); ) {
        // a vector to store data_points
        std::vector<double> datapoint_vector;

        // split string by delimeter
        auto start = 0U;
        auto end = line.find(delimeter);

        // get through a line
        while (end != std::string::npos) {
            datapoint_vector.push_back(std::stod(line.substr(start, end - start)));
            start = end + delimeter.length();
            end = line.find(delimeter, start);
        }
        datapoint_vector.push_back(std::stod(line.substr(start, end)));
        data_batch.push_back(datapoint_vector);
    }

    arma::mat data_mat = arma::zeros<arma::mat>(data_batch.size(), data_batch[0].size());

    int n = data_batch.size();
    for (int i = 0; i < n; i++) {
        arma::mat r(data_batch[i]);
        data_mat.row(i) = r.t();
    }
    return data_mat;
}

/**
 * Main function
 *
 * @param argc: num. arguments defined by users
 * @param argv: argument values by users
 * @return 0
 */
int main(int argc, char* argv[])
{
    // Path to the dataset used for training and testing.
    std::string datasetPath;
    if (argc < 2) {
        std::cout << "Usage: ./lr_regression <path_to_dataset_file>" << std::endl;
        exit(1);
    } else
        datasetPath = argv[1];

    // File for saving the model.
    const std::string modelFile = "./linear_regressor.bin";

    // Delare dataset and result variables
    arma::mat dataset;
    arma::vec responses; // The responses, one row for each row in data.

    // In Armadillo rows represent features, columns represent data points.
    std::cout << "------------------------------------\n";
    std::cout << "1. Reading data." << std::endl;
    std::cout << "------------------------------------\n";
    bool loadedDataset = data::Load(datasetPath, dataset);
    if (!loadedDataset)
        return -1;
    std::cout << "\tSize of the whole dataset: " << dataset.n_rows << "x" << dataset.n_cols << std::endl;

    // Split the dataset for training and testing
    arma::mat trainData, validData;
    const int RATIO = 0.3;  // 30% for training
    const int T_FROM_ROW = 0, T_TO_ROW = 6;
    const int T_FROM_COL = 0, T_TO_COL = 49;
    const int V_FROM_ROW = 0, V_TO_ROW = 6;
    const int V_FROM_COL = 50, V_TO_COL = 193;
    trainData = dataset.submat(T_FROM_ROW, T_FROM_COL, T_TO_ROW, T_TO_COL);
    validData = dataset.submat(V_FROM_ROW, V_FROM_COL, V_TO_ROW, V_TO_COL);
    std::cout << "\tChecking dataset: "
            << "Train=" << trainData.n_rows << "x" << trainData.n_cols
            << ", Test=" << validData.n_rows << "x" << validData.n_cols
            << std::endl;

    // std::cout << "------------------------------" << std::endl;
    // std::cout << trainData.row(0) << std::endl;
    // std::cout << validData.row(0) << std::endl;
    // std::cout << "------------------------------" << std::endl;

    // The train and valid datasets contain both - the features as well as the
    // prediction. Split these into separate matrices.
    arma::mat trainX =
        trainData.submat(0, 0, trainData.n_rows-2, trainData.n_cols-1);
    arma::mat validX =
        validData.submat(0, 0, validData.n_rows-2, validData.n_cols-1);
    std::cout << "\t   Train_X=" << trainX.n_rows << "x" << trainX.n_cols
            << ", Test_X=" << validX.n_rows << "x" << validX.n_cols
            << std::endl;

    // Create prediction data for training and validatiion datasets.
    arma::rowvec trainY = trainData.row(trainData.n_rows-1);
    arma::mat validY = validData.row(validData.n_rows-1);
    std::cout << "\t   Train_Y=" << trainY.n_rows << "x" << trainY.n_cols
            << ", Test_Y=" << validY.n_rows << "x" << validY.n_cols
            << std::endl;
    
    // Showing trainY, validY
    // std::cout << "---------------------------------" << std::endl;
    // std::cout << "Trying to show trainY, validY:" << std::endl;
    // std::cout << trainY << std::endl;
    // std::cout << validY << std::endl;

    // ----------------------------------------------------------
    // 1. Linear Regression
    // ----------------------------------------------------------
    mlpack::regression::LinearRegression lr(trainX, trainY);

    // ----------------------------------------------------------
    // 2. Ridge Regression: lambda = 0.2
    // ----------------------------------------------------------
    const double lambda = 0.2;
    mlpack::regression::LinearRegression rr(trainX, trainY, lambda);

    // ----------------------------------------------------------
    // 3. LARS:  a stage-wise homotopy-based algorithm for l1-regularized linear regression (LASSO)
    // and l1+l2 regularized linear regression (Elastic Net)
    // ----------------------------------------------------------
    mlpack::regression::LARS lars(trainX, trainY);

    // ----------------------------------------------------------
    // 4. Bayesian Linear Regression
    // ----------------------------------------------------------
    mlpack::regression::BayesianLinearRegression blr;
    blr.Train(trainX, trainY);

    // ----------------------------------------------------------
    // 5. SVM
    // ----------------------------------------------------------
    // const double lambda = 0.005;
    // const double delta  = 0.5;
    // int num_classes = 5;
    // mlpack::svm::LinearSVM<> lr(6, num_classes, lambda, delta);
    // lr.Train(trainX, trainY, num_classes);
    // arma::rowvec pred_values;
    // lr.Classify(validX, pred_values);

    // Get the parameters, or coefficients.
    // arma::vec parameters = lr.Parameters();

    // Evaluate the predictions
    arma::rowvec pred_values_lr;
    arma::rowvec pred_values_rr;
    arma::rowvec pred_values_lars;
    arma::rowvec pred_values_blr;

    lr.Predict(validX, pred_values_lr);
    rr.Predict(validX, pred_values_rr);
    lars.Predict(validX, pred_values_lars);
    blr.Predict(validX, pred_values_blr);

    std::cout << "------------------------------------\n";
    std::cout << "2. Checking the prediction model....\n";
    std::cout << "------------------------------------\n";
    // std::cout << "Predict | Groundtruth\n";
    // char buf[1024];
    // for (int i = 0; i < pred_values.size(); i++){
    //     sprintf(buf, "%7.4f | %7.4f", pred_values[i], validY[i]);
    //     std::cout << buf << std::endl;
    // }


    // Try with a single value
    // arma::rowvec single_value;
    // const int col_idx = 5;
    // lr.Predict(validX.col(col_idx), single_value);
    // std::cout << "3. Checking with a single input." << std::endl;
    // std::cout << "---------------------------------------" << std::endl;
    // std::cout << "\t Input: " << validX.col(col_idx).t() << std::endl;
    // std::cout << "\t Groundtruth: " << validY.col(col_idx) << std::endl;
    // std::cout << "\t Prediction: " << single_value << std::endl;

    return 0;
}