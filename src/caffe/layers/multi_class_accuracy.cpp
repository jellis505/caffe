#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiClassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultiClassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
  top[2]->Reshape(top_shape);
  top[3]->Reshape(top_shape);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  this->threshold_ = this->layer_param_.accuracy_param().bag_threshold();
}

template <typename Dtype>
void MultiClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int total_count = bottom[0]->count();
  int correct = 0;
  double pos_correct = 0;
  double curr_label = 0;
  double curr_pred = 0;
  int num_total_pos = 0;
  int num_pos_predictions = 0;
  for (int i = 0; i < total_count; i++) {
    if (bottom_data[i] > this->threshold_) {
      curr_pred = 1.0;
    } else {
      curr_pred = 0.0;
    }
    curr_label = bottom_label[i];
    num_pos_predictions += curr_pred;
    if (curr_label == curr_pred) {
      correct++;
    }
    if (curr_label == 1.0) {
      num_total_pos++;
      if (curr_pred == 1.0) {
        pos_correct++;
      }
    }
  }
  // Output to the data sets
  top[0]->mutable_cpu_data()[0] = num_total_pos;
  top[1]->mutable_cpu_data()[0] = num_pos_predictions;
  top[2]->mutable_cpu_data()[0] = correct / (double)total_count;
  top[3]->mutable_cpu_data()[0] = pos_correct / (double)num_total_pos;
}

INSTANTIATE_CLASS(MultiClassAccuracyLayer);
REGISTER_LAYER_CLASS(MultiClassAccuracy);

}  // namespace caffe
