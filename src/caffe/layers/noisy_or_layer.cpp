#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NoisyOrLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  // This is the number of classes that we have
  const int num_output = this->layer_param_.noisy_or_param().num_output();
  output_size_ = num_output;
  const int num_instances = this->layer_param_.noisy_or_param().num_instances();
  num_instances_ = num_instances;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.noisy_or_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  mult_length_vector_ = bottom[0]->count(axis);

  // Check to make sure that the length of bottom and top probability layers are the same
  CHECK_EQ(mult_length_vector_, output_size_);
  // TODO: Add a check here for the size of the output as well.
  // Check if we need to set up the blob weights...
  // I don't think that this layer actually needs to set up the blobs at all.

  // Now let's set up the top vector size
  static const int top_shape_array[] = {1, 1, 1, output_size_};
  vector<int> top_shape(top_shape_array, top_shape_array + 4);
  top[0]->Reshape(top_shape);
  mutable_bottom_ = bottom[0];
}

template <typename Dtype>
void NoisyOrLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  // This is the number of classes that we have
  const int num_output = this->layer_param_.noisy_or_param().num_output();
  output_size_ = num_output;
  const int num_instances = this->layer_param_.noisy_or_param().num_instances();
  num_instances_ = num_instances;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.noisy_or_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  mult_length_vector_ = bottom[0]->count(axis);

  // Check to make sure that the length of bottom and top probability layers are the same
  CHECK_EQ(mult_length_vector_, output_size_);
  // TODO: Add a check here for the size of the output as well.
  // Check if we need to set up the blob weights...
  // I don't think that this layer actually needs to set up the blobs at all.

  // Now let's set up the top vector size
  static const int top_shape_array[] = {1, 1, 1, output_size_};
  vector<int> top_shape(top_shape_array, top_shape_array + 4);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void NoisyOrLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Simple for loop implementation
  // Create negative of the bottom data
  //caffe_scal(bottom_count, (Dtype)-1., bottom_data);
  
  // Calculate (1-x) for ever value in our blob
  //caffe_add_scalar(bottom_count, (Dtype)1., bottom_data);

  //Now let's calculate the value of a single layer
  for (int i = 0; i < output_size_; i++) {
    double p_bag_given_instances = 1;
    for (int j = 0; j < num_instances_; j++) {
      //if (i == 0 and j == 0) {
      //  LOG(INFO) << "Sigmoid Layer " << bottom[0]->cpu_data()[0];
      //}
      p_bag_given_instances = p_bag_given_instances * (1 - bottom[0]->data_at(j,0,0,i));
    }
    // This top_data should have only one dimension, therefore we can access it directly
    //if (i == 0) {
    //  LOG(INFO) << "P_bag_given_instances " << (1 - p_bag_given_instances);
    //}
    top_data[i] = (1 - p_bag_given_instances);
  }
}

template <typename Dtype>
void NoisyOrLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Because there are no tunable weights on this layer, this simply amounts to a 
  // weighting of the examples within the bag as to which ones the errors are propgated
  // down to.
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff(); // This represents the bag difference
    // Weighting of the particular examples
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int index = 0;
    Dtype eps = 0.0001;
    for (int i = 0; i < output_size_; i++) {
      for (int j = 0; j < num_instances_; j++) {
        // This returns the index for accessing our layers
        index = j* output_size_ + i;
        bottom_diff[index] = -(((bottom[0]->cpu_data()[index] + eps)  * top_diff[i]));

        //if (top_diff[i] > 0.01) {
        //  LOG(INFO) << "Value Num: " << i;
        //  LOG(INFO) << "Bag Weight: " << top_diff[i];
        //  LOG(INFO) << "Instance Weight: " << bottom[0]->data_at(j,0,0,i);
        //  LOG(INFO) << "Total Error Propogation: " << bottom_diff[index]; 
        //}

        //if (i == 0 and j == 0) {
        //  LOG(INFO) << "Noisy-Or Bag Weight " << top_diff[i];
        //  LOG(INFO) << "Instance Weight " << bottom[0]->data_at(j, 0, 0, i);
        //  LOG(INFO) << "Diff " << bottom[0]->diff_at(j, 0, 0, i);
        //}
      }
    }
  }
}

// Include when we have GPU implementation
//#ifdef CPU_ONLY
//STUB_GPU(NoisyOrLayer);
//#endif

INSTANTIATE_CLASS(NoisyOrLayer);
REGISTER_LAYER_CLASS(NoisyOr);

}  // namespace caffe

