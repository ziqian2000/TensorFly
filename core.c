#include <stdio.h>

int conv2d( float* input,	int input_batch,	int input_h,	int input_w,	int input_c,
			float* filter,	int filter_h,		int filter_w,	int filter_c,	int filter_o_c,
			float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
			int stride_h, 	int stride_w)
{
	/*
		When I changed the order of enumetation, it becomes faster ...
		But I don't know why
	*/
	int batch_h_size = input_w * input_c * stride_h;
	int batch_h_w_size = input_c * stride_w;
	int filter_h_size = filter_w * filter_c * filter_o_c;
	int filter_h_w_size = filter_c * filter_o_c;
	int batch_size = input_h * input_w * input_c;
	float* ptr_output = output;
	float* ptr_batch = input;
	for(int batch = 0; batch < output_batch; ++batch) // for each batch in output
	{
		float* ptr_batch_h = ptr_batch;
		for(int o_h = 0; o_h < output_h; ++o_h) // for each row in output
		{
			float* ptr_batch_h_w = ptr_batch_h;
			for(int o_w = 0; o_w < output_w; ++o_w) // for each column in output
			{
				/*convolution begin*/
				float *ptr_filter_h = filter;
				float *ptr_batch_h_w_h2 = ptr_batch_h_w;
				for(int f_h = 0; f_h < filter_h; ++f_h)
				{
					float *ptr_filter_h_w = ptr_filter_h;
					float *ptr_batch_h_w_h2_w2 = ptr_batch_h_w_h2;
					for(int f_w = 0; f_w < filter_w; ++f_w)
					{
						float *ptr_filter_h_w_c = ptr_filter_h_w;
						float *ptr_batch_h_w_h2_w2_c = ptr_batch_h_w_h2_w2;
						for(int f_c = 0; f_c < filter_c; ++f_c)
						{
							for(int o_c = 0; o_c < output_c; ++o_c) // for each channel in output
							{
								ptr_output[o_c] += ptr_batch_h_w_h2_w2_c[f_c] * (ptr_filter_h_w_c[o_c]); // calc
							}
							ptr_filter_h_w_c += filter_o_c;
						}
						ptr_filter_h_w += filter_h_w_size;
						ptr_batch_h_w_h2_w2 += batch_h_w_size;
					}
					ptr_filter_h += filter_h_size;
					ptr_batch_h_w_h2 += batch_h_size;

					/*convolution end*/
				}
				ptr_output += output_c;
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
	int input_batch_size = input_h * input_w * input_c;
	int grad_batch_size = grad_h * grad_w * grad_c;
	int grad_batch_h_size = grad_w * grad_c;
	int output_h_size = output_w * output_c * output_o_c;
	int output_h_w_size = output_c * output_o_c;

	float *ptr_input_batch = input;
	float *ptr_grad_batch = grad;
	for(int batch = 0; batch < input_batch; batch++)
	{
		float* ptr_input_batch_h = ptr_input_batch;
		float* ptr_output_h = output;
		for(int o_h = 0; o_h < output_h; o_h++)
		{
			float *ptr_input_batch_h_w = ptr_input_batch_h;
			float *ptr_output_h_w = ptr_output_h;
			for(int o_w = 0; o_w < output_w; o_w++)
			{

				float *ptr_input_batch_h_w_h2 = ptr_input_batch_h_w;
				float *ptr_grad_batch_h = ptr_grad_batch;

				for(int i_h = 0; i_h < grad_h; i_h++)
				{
					float *ptr_input_batch_h_w_h2_w2 = ptr_input_batch_h_w_h2;
					float *ptr_grad_batch_h_w = ptr_grad_batch_h;

					for(int i_w = 0; i_w < grad_w; i_w++)
					{
						float *ptr_output_h_w_c = ptr_output_h_w;
						for(int o_c = 0; o_c < output_c; o_c++)
						{
							for(int o_o_c = 0; o_o_c < output_o_c; o_o_c++)
							{
								ptr_output_h_w_c[o_o_c] += ptr_input_batch_h_w_h2_w2[o_c] * ptr_grad_batch_h_w[o_o_c];
							}
							ptr_output_h_w_c += output_o_c;
						}
						ptr_input_batch_h_w_h2_w2 += input_c;
						ptr_grad_batch_h_w += grad_c;
					}
					ptr_input_batch_h_w_h2 += input_batch_h_size;
					ptr_grad_batch_h += grad_batch_h_size;
				}
				ptr_output_h_w += output_h_w_size;
				ptr_input_batch_h_w += input_c;
			}
			ptr_output_h += output_h_size;
			ptr_input_batch_h += input_batch_h_size;
		}
		ptr_input_batch += input_batch_size;
		ptr_grad_batch += grad_batch_size;
	}
	return 0;
}

int maxpool_grad(   float* input,	int input_batch,	int input_h,	int input_w,	int input_c,
			 		float* grad,	int grad_batch,		int grad_h,		int grad_w,		int grad_c,
			 		float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
			 		int ksize_h, 	int ksize_w,		int stride_h, 	int stride_w)
{
	
	int input_batch_size = input_h * input_w * input_c;
	int input_batch_h_size = input_w * input_c;
	int grad_batch_size = grad_h * grad_w * grad_c;
	int grad_batch_h_size = grad_w * grad_c;

	float *ptr_input_batch = input;
	float *ptr_grad_batch = grad;

	for(int batch = 0; batch < input_batch; ++batch)
	{
		float *ptr_input_batch_h = ptr_input_batch;
		float *ptr_grad_batch_h = ptr_grad_batch;
		for(int g_h = 0; g_h < grad_h; ++g_h)
		{
			float *ptr_input_batch_h_w = ptr_input_batch_h;
			float *ptr_grad_batch_h_w = ptr_grad_batch_h;
			for(int g_w = 0; g_w < grad_w; ++g_w)
			{
				float *ptr_input_batch_h_w_c = ptr_input_batch_h_w;
				float *ptr_grad_batch_h_w_c = ptr_grad_batch_h_w;
				for(int c = 0; c < input_c; ++c)
				{
					/* find the position */
					float *ptr_input_batch_h_w_c_h2 = ptr_input_batch_h_w_c;
					float *pos = ptr_input_batch_h_w_c;
					for(int h = 0; h < ksize_h; ++h)
					{
						float *ptr_input_batch_h_w_c_h2_w2 = ptr_input_batch_h_w_c_h2;
						for(int w = 0; w < ksize_w; ++w)
						{
							if(*ptr_input_batch_h_w_c_h2_w2 > *pos)
								pos = ptr_input_batch_h_w_c_h2_w2;
							ptr_input_batch_h_w_c_h2_w2 += input_c;
						}
						ptr_input_batch_h_w_c_h2 += input_batch_h_size;
					}
					/* find the position */
					output[pos - input] += *ptr_grad_batch_h_w_c;
					++ptr_input_batch_h_w_c;
					++ptr_grad_batch_h_w_c;
				}
				ptr_input_batch_h_w += input_c * stride_w;
				ptr_grad_batch_h_w += grad_c;
			}
			ptr_input_batch_h += input_batch_h_size * stride_h;
			ptr_grad_batch_h += grad_batch_h_size;
		}
		ptr_input_batch += input_batch_size;
		ptr_grad_batch += grad_batch_size;
	}

	return 0;
}