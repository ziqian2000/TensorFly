#include <stdio.h>

int conv2d( float* input,	int input_batch,	int input_h,	int input_w,	int input_c,
			float* filter,	int filter_h,		int filter_w,	int filter_c,	int filter_o_c,
			float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
			int stride_h, 	int stride_w)
{
	float* ptr_output = output;
	float* ptr_batch = input;
	int batch_size = input_h * input_w * input_c;
	for(int batch = 0; batch < output_batch; batch++) // for each batch in output
	{
		float* ptr_batch_h = ptr_batch;
		int batch_h_size = input_w * input_c * stride_h;
		for(int o_h = 0; o_h < output_h; o_h++) // for each row in output
		{
			float* ptr_batch_h_w = ptr_batch_h;
			int batch_h_w_size = input_c * stride_w;
			for(int o_w = 0; o_w < output_w; o_w++) // for each column in output
			{
				for(int o_c = 0; o_c < output_c; o_c++) // for each channel in output
				{
					/*convolution begin*/
					float *ptr_filter_h = filter;
					int filter_h_size = filter_w * filter_c * filter_o_c;
					float *ptr_batch_h_w_h2 = ptr_batch_h_w;
					for(int f_h = 0; f_h < filter_h; f_h++)
					{
						float *ptr_filter_h_w = ptr_filter_h;
						int filter_h_w_size = filter_c * filter_o_c;
						float *ptr_batch_h_w_h2_w2 = ptr_batch_h_w_h2;
						for(int f_w = 0; f_w < filter_w; f_w++)
						{
							float *ptr_filter_h_w_c = ptr_filter_h_w;
							int filter_h_w_c_size = filter_o_c;
							float *ptr_batch_h_w_h2_w2_c = ptr_batch_h_w_h2_w2;
							for(int f_c = 0; f_c < filter_c; f_c++)
							{
								float *ptr_filter_element = ptr_filter_h_w_c + o_c;
								(*ptr_output) += (*ptr_batch_h_w_h2_w2_c) * (*ptr_filter_element); // calc
								ptr_filter_h_w_c += filter_h_w_c_size;
								ptr_batch_h_w_h2_w2_c++;
							}
							ptr_filter_h_w += filter_h_w_size;
							ptr_batch_h_w_h2_w2 += batch_h_w_size;
						}
						ptr_filter_h += filter_h_size;
						ptr_batch_h_w_h2 += batch_h_size;
					}
					/*convolution end*/

					ptr_output++;
				}
				ptr_batch_h_w += batch_h_w_size;
			}
			ptr_batch_h += batch_h_size;
		}
		ptr_batch += batch_size;
	}
	return 0;
}

int conv2d_grad(float* input,	int input_batch,	int input_h,	int input_w,	int input_c,
				float* grad,	int grad_batch,		int grad_h,		int grad_w,		int grad_c,
				float* output,	int output_h,		int output_w,	int output_c,	int output_o_c)
{
	int input_batch_h_size = input_w * input_c;
	int grad_batch_h_size = grad_w * grad_c;
	float* ptr_output = output;

	// for(int i = 0; i < input_batch * input_h * input_w * input_c; i++) printf("%lf ", input[i]); puts("");
	// for(int i = 0; i < grad_batch * grad_h * grad_w * grad_c; i++) printf("%lf ", grad[i]); puts("");
	// for(int i = 0; i < output_h * output_w * output_c * output_o_c; i++) printf("%lf ", output[i]); puts("");

	for(int o_h = 0; o_h < output_h; o_h++)
	{
		for(int o_w = 0; o_w < output_w; o_w++)
		{
			for(int o_c = 0; o_c < output_c; o_c++)
			{
				for(int o_o_c = 0; o_o_c < output_o_c; o_o_c++)
				{
					/*calculation begin*/
					float *ptr_input_batch = input;
					int input_batch_size = input_h * input_w * input_c;

					float *ptr_grad_batch = grad;
					int grad_batch_size = grad_h * grad_w * grad_c;

					for(int batch = 0; batch < input_batch; batch++)
					{
						float *ptr_input_batch_h_w = ptr_input_batch + o_h * input_batch_h_size + o_w * input_c;
						float *ptr_input_batch_h_w_h2 = ptr_input_batch_h_w;

						float *ptr_grad_batch_h = ptr_grad_batch;

						for(int i_h = 0; i_h < grad_h; i_h++)
						{
							float *ptr_input_batch_h_w_h2_w2 = ptr_input_batch_h_w_h2;

							float *ptr_grad_batch_h_w = ptr_grad_batch_h;
							for(int i_w = 0; i_w < grad_w; i_w++)
							{
								*ptr_output += (*(ptr_input_batch_h_w_h2_w2 + o_c)) * (*(ptr_grad_batch_h_w + o_o_c));
								ptr_input_batch_h_w_h2_w2 += input_c;
								ptr_grad_batch_h_w += grad_c;
							}
							ptr_input_batch_h_w_h2 += input_batch_h_size;
							ptr_grad_batch_h += grad_batch_h_size;
						}

						ptr_input_batch += input_batch_size;
						ptr_grad_batch += grad_batch_size;
					}
					/*calculation end*/
					ptr_output++;
				}
			}
		}
	}
	return 0;
}