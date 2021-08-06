#include <iostream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

REGISTER_OP("ZeroCounter")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("message: string = ''")
    .Attr("skip_threshold: float = 0");

using namespace tensorflow;

class ZeroCounterOp : public OpKernel {
    public:
        explicit ZeroCounterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("skip_threshold", &skip_threshold_));
        }

        void Compute(OpKernelContext* ctx) override {
            if (IsRefType(ctx->input_dtype(0))) {
                ctx->forward_ref_input_to_ref_output(0, 0);
            } else {
                ctx->set_output(0, ctx->input(0));
            }

            const Tensor &input_tensor = ctx->input(0);
            int last_dim, other_dim = 1;
            std::vector<int> shape;
            int num_dimensions = input_tensor.shape().dims();
            for(int ii_dim=0; ii_dim<num_dimensions; ii_dim++) {
                shape.push_back(input_tensor.shape().dim_size(ii_dim));
                if (ii_dim == (num_dimensions - 1)) {
                    last_dim = input_tensor.shape().dim_size(ii_dim);
                } else {
                    other_dim *= input_tensor.shape().dim_size(ii_dim);
                }
            }

            //std::cerr << "[DEBUG] input vector's shape : (";
            //for(std::vector<int>::iterator iter = shape.begin(); iter != shape.end(); iter++){
                //std::cout << *iter << ",";
            //}
            //std::cerr << ")" << std::endl;

            //std::cerr << "[DEBUG] other_dim: " << other_dim << ", last_dim: " << last_dim << std::endl;

            // start reshape
            Tensor new_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({other_dim, last_dim}));
            if (!new_input.CopyFrom(input_tensor, tensorflow::TensorShape({other_dim, last_dim}))) {
                std::cerr << "Unsuccessfully reshaped image" << std::endl;
                return;
            }

            // Count zeros
            string msg;
            strings::StrAppend(&msg, message_);
            msg += std::string(" => ");
            auto new_input_mapped = new_input.tensor<float, 2>();
            for (int i = 0; i < other_dim; i++) {
                int zero_count = 0;
                for (int j = 0; j < last_dim; j++) {
                    if (new_input_mapped(i,j) < skip_threshold_) {
                        zero_count++;
                    }
                }
                msg += std::to_string(zero_count);
                msg += std::string(",");
            }
            std::cerr << msg << std::endl;
        }
    private:
        string message_;
        float skip_threshold_;
};

REGISTER_KERNEL_BUILDER(Name("ZeroCounter").Device(DEVICE_CPU), ZeroCounterOp);
