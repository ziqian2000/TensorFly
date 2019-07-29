#include <cstdio>
#include <cstring>
#include <cblas.h>

extern "C"
inline int matmul(float* matA, float* matB, float* matC, int n, int k, int m, float alpha = 1.0, float beta = 0.0)
{
	/*
		 cblas_sgemm(order,transA,transB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);
		 alpha =1,beta =0 的情况下，等于两个矩阵相乘。
		 第一参数 oreder 候选值 有ClasRowMajow 和ClasColMajow 这两个参数决定一维数组怎样存储在内存中,一般用ClasRowMajow
		 参数 transA和transB ：表示矩阵A，B是否进行转置。候选参数 CblasTrans 和CblasNoTrans.
		 参数M：表示 A或C的行数。如果A转置，则表示转置后的行数
		 参数N：表示 B或C的列数。如果B转置，则表示转置后的列数。
		 参数K：表示 A的列数或B的行数（A的列数=B的行数）。如果A转置，则表示转置后的列数。
		 参数LDA：表示A的列数，与转置与否无关。
		 参数LDB：表示B的列数，与转置与否无关。
		 参数LDC：始终=N
	*/
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, k, alpha, matA, k, matB, m, beta, matC, m);
    return 0;
}

extern "C"
int conv2d( float* input,	int input_batch,	int input_h,	int input_w,	int input_c,
			float* filter,	int filter_h,		int filter_w,	int filter_c,	int filter_o_c,
			float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
			int stride_h, 	int stride_w)
{
	
	int batch_size = input_h * input_w * input_c;
	int batch_h_size = input_w * input_c;
	int batch_h_w_size = input_c;
	int output_batch_size = output_h * output_w * output_c;

	int move_size = filter_w * input_c;
	int copy_size = move_size * sizeof(float);

	int dim1 = output_h * output_w;
	int dim2 = filter_h * filter_w * filter_c;
	int dim3 = filter_o_c;

	float *ptr_input_batch = input;
	float *ptr_output_batch = output;
	float *img = new float[dim1 * dim2]; // temporary vector for calculation

	for(int batch = 0; batch < input_batch; batch++, ptr_input_batch += batch_size, ptr_output_batch += output_batch_size) // for each patch
	{
		float *ptr_img = img;
		float *ptr_input_batch_h = ptr_input_batch;
		for(int i_h = 0; i_h < output_h; i_h ++, ptr_input_batch_h += batch_h_size * stride_h)
		{
			float *ptr_input_batch_h_w = ptr_input_batch_h;
			for(int i_w = 0; i_w < output_w; i_w ++, ptr_input_batch_h_w += batch_h_w_size * stride_w)
			{
				float *ptr_input_batch_h_w_h2 = ptr_input_batch_h_w;
				for(int i_h2 = 0; i_h2 < filter_h; i_h2++, ptr_input_batch_h_w_h2 += batch_h_size)
				{
					memcpy(ptr_img, ptr_input_batch_h_w_h2, copy_size);
					ptr_img += move_size;
				}
			}
		}
		matmul(img, filter, ptr_output_batch, dim1, dim2, dim3);
	}
	delete []img;
	return 0;
}

extern "C"
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

extern "C"
int maxpool(float* input,	int input_batch,	int input_h,	int input_w,	int input_c,
	 		float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
	 		float* pos,		int ksize_h,	 	int ksize_w,	int stride_h, 	int stride_w,
	 		int up, 		int left)
{
	int input_batch_size = input_h * input_w * input_c;
	int input_batch_h_size = input_w * input_c;
	int output_batch_size = output_h * output_w * output_c;
	int output_batch_h_size = output_w * output_c;
	int idx = 0;
	float *ptr_input_batch = input;
	float *ptr_output_batch = output;
	float *ptr_pos_batch = pos;
	memset(output, 254, sizeof(float) * output_batch * output_batch_size);
	for(int batch = 0; batch < input_batch; batch++, ptr_input_batch += input_batch_size, 
													 ptr_output_batch += output_batch_size, 
													 ptr_pos_batch += output_batch_size)
	{
		for(int _h = 0; _h < input_h; _h++)
		{
			for(int _w = 0; _w < input_w; _w++)
			{
				int h = _h + up, w = _w + left;

				++idx;

				float *ptr_input_cur = ptr_input_batch + h * input_batch_h_size + w * input_c;
				float *ptr_output_cur = ptr_output_batch + h / stride_h * output_batch_h_size + w / stride_w * output_c;
				float *ptr_pos_cur = ptr_pos_batch + h / stride_h * output_batch_h_size + w / stride_w * output_c;

				for(int c = 0; c < input_c; c++)
				{
					if(*ptr_input_cur > *ptr_output_cur)
					{
						*ptr_output_cur	= *ptr_input_cur;
						*ptr_pos_cur = idx;
					}
					ptr_input_cur++;
					ptr_output_cur++;
					ptr_pos_cur++;
				}
			}
		}
	}
	return 0;
}

extern "C"
int maxpool_grad(   float* pos,		int pos_batch,		int pos_h,		int pos_w,		int pos_c,
			 		float* grad,	int grad_batch,		int grad_h,		int grad_w,		int grad_c,
			 		float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
	 				int ksize_h,	int ksize_w,		int stride_h, 	int stride_w)
{
	int grad_batch_size = grad_h * grad_w * grad_c;
	int grad_batch_h_size = grad_w * grad_c;
	int output_batch_size = output_h * output_w * output_c;
	int output_batch_h_size = output_w * output_c;

	int idx = 0;

	float *ptr_pos_batch = pos;
	float *ptr_grad_batch = grad;
	float *ptr_output_batch = output;

	for(int batch = 0; batch < output_batch; batch++, ptr_grad_batch += grad_batch_size, 
													  ptr_pos_batch += grad_batch_size,
													  ptr_output_batch += output_batch_size)
	{
		for(int h = 0; h < output_h; h++)
		{
			for(int w = 0; w < output_w; w++)
			{
				++idx;

				float *ptr_grad_cur = ptr_grad_batch + h / stride_h * grad_batch_h_size + w / stride_w * grad_c;
				float *ptr_pos_cur = ptr_pos_batch + h / stride_h * grad_batch_h_size + w / stride_w * grad_c;
				float *ptr_output_cur = ptr_output_batch + h * output_batch_h_size + w * output_c;

				for(int c = 0; c < output_c; c++)
				{
					if(*ptr_pos_cur == idx)
					{
						*ptr_output_cur += *ptr_grad_cur;
					}
					ptr_grad_cur++;
					ptr_output_cur++;
					ptr_pos_cur++;
				}
			}
		}
	}
	return 0;
}