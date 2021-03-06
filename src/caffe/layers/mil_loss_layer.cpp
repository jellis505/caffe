#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MilLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void MilLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void MilLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  Dtype positive_count = 0.0;
  for (int i = 0; i < count; ++i) {
    positive_count += target[i];
    loss -= target[i] * log(input_data[i] + (target[i] == Dtype(0)));
    loss -= (1 - target[i]) * log(1 - input_data[i] + (target[i] == Dtype(1)));
  }
  top[0]->mutable_cpu_data()[0] = (loss / num) ;
}

template <typename Dtype>
void MilLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* output_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const Dtype eps = 0.00001;
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; i++) {
      Dtype denom = (output_data[i] * ((Dtype) 1.0 - output_data[i]));
      Dtype used_denom = std::max(denom, eps);
      bottom_diff[i] = (output_data[i] - target[i]) / used_denom;
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //const Dtype loss_weight = 1.0;
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

//#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(MilLossLayer, Backward);
//#endif

INSTANTIATE_CLASS(MilLossLayer);
REGISTER_LAYER_CLASS(MilLoss);

}  // namespace caffe
