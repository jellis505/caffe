#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class NoisyOrLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  NoisyOrLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 1, 1, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(.5);
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    LOG(INFO) << "Filler Value" << filler_param.value();
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);

    // Now set up the errors for test propogation
    true_labels_.push_back(1.);
    true_labels_.push_back(0.);
    true_labels_.push_back(1.);

    prop_down_.push_back(true);

    // Now add the vector example top layer
  }
  virtual ~NoisyOrLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* blob_bottom_;
  Blob<Dtype>* blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> prop_down_;
  vector<double> true_labels_;
};

TYPED_TEST_CASE(NoisyOrLayerTest, TestDtypesAndDevices);

TYPED_TEST(NoisyOrLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  NoisyOrParameter* noisy_or_param =
      layer_param.mutable_noisy_or_param();
  noisy_or_param->set_num_output(3);
  noisy_or_param->set_num_instances(2);
  shared_ptr<NoisyOrLayer<Dtype> > layer(
      new NoisyOrLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 1);
}

TYPED_TEST(NoisyOrLayerTest, ForwardTest) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    NoisyOrParameter* noisy_or_param =
      layer_param.mutable_noisy_or_param();
    noisy_or_param->set_num_output(3);
    noisy_or_param->set_num_instances(2);
    shared_ptr<NoisyOrLayer<Dtype> > layer(new NoisyOrLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    const int num = this->blob_bottom_vec_[0]->num();
    for (int i = 0; i < count; ++i) {
      Dtype value = 1.0;
      for (int j = 0; j < num; j++) {
        value = value * ((Dtype)1.0 - (Dtype)this->blob_bottom_vec_[0]->data_at(j, 0, 0, i));
      }
      EXPECT_EQ((Dtype) (1.0 - value), data[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

/*
TYPED_TEST(NoisyOrLayerTest, TestErrorProp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    NoisyOrParameter* noisy_or_param =
      layer_param.mutable_noisy_or_param();
    noisy_or_param->set_num_output(3);
    noisy_or_param->set_num_instances(2);
    shared_ptr<NoisyOrLayer<Dtype> > layer(new NoisyOrLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Calculate Loss from above, it is the (true label - bag_label) / bag_label
    const int count = this->blob_top_vec_[0]->count();
    const Dtype* data = this->blob_top_vec_[0]->cpu_data();
    for (int i = 0; i < count; i++) {;
      this->blob_top_vec_[0]->mutable_cpu_diff()[i] = (this->true_labels_[i] - data[i]) / (data[i]);
    }
    layer->Backward(this->blob_top_vec_, this->prop_down_, this->blob_bottom_vec_);
    for (int i = 0; i < noisy_or_param->num_output(); i++) {
      for (int j = 0; j < noisy_or_param->num_instances(); j++) {
        if (i == 1) {
          EXPECT_EQ(-0.5, this->blob_bottom_vec_[0]->diff_at(j,0,0,i));
        } else {
          EXPECT_LE(.166, this->blob_bottom_vec_[0]->diff_at(j,0,0,i));
          EXPECT_GE(.167, this->blob_bottom_vec_[0]->diff_at(j,0,0,i));
        }
      }
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
*/

TYPED_TEST(NoisyOrLayerTest, GradientChecker) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    NoisyOrParameter* noisy_or_param =
      layer_param.mutable_noisy_or_param();
    noisy_or_param->set_num_output(3);
    noisy_or_param->set_num_instances(2);
    NoisyOrLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


} // namespace caffe
