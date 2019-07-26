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