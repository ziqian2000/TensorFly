#include <cblas.h>
#include <stdlib.h>
#include <string.h>

float *mem = NULL;
int mem_len;

float pos_buffer[5][1000000]; // big enough
int pos_timer = -1;


inline float swap_f(float *a, float *b){float c = *a; *a = *b; *b = c;}
inline int swap_i(int *a, int *b){int c = *a; *a = *b; *b = c;}

inline int min(int a, int b){return a < b ? a : b;}
inline int max(int a, int b){return a > b ? a : b;}

inline int matmul(float* matA, float* matB, float* matC, int n, int k, int m, float beta)
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
	float alpha = 1.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, k, alpha, matA, k, matB, m, beta, matC, m);
    return 0;
}


void matmul_trans(float* a, float* b, float* c, int na, int ma, int nb, int mb, int transA, int transB)
{
    if(transA)
        transB
            ? cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, ma, nb, mb, 1.0, a, ma, b, mb, 0.0, c, nb)
            : cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, ma, mb, na, 1.0, a, ma, b, mb, 0.0, c, mb);
    else
        transB
            ? cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, na, nb, mb, 1.0, a, ma, b, mb, 0.0, c, nb)
            : cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, na, mb, ma, 1.0, a, ma, b, mb, 0.0, c, mb);
}

inline float* swap_dim2_and_dim3(float *filter, int filter_h, int filter_w, int filter_c, int filter_o_c)
{
	if(mem_len < filter_h * filter_w * filter_c * filter_o_c)
	{
		free(mem);
		mem = (float*)malloc((mem_len = filter_h * filter_w * filter_c * filter_o_c) * sizeof(float));
	}
	float *tmp = mem;
	for(int h = 0; h < filter_h; h++)
	{
		for(int w = 0; w < filter_w; w++)
		{
			float *tmp_h_w = 	tmp + 	 	h * filter_w * filter_c * filter_o_c +		w * filter_c * filter_o_c;
			float *filter_h_w = filter + 	h * filter_w * filter_c * filter_o_c +		w * filter_c * filter_o_c;
			for(int c = 0; c < filter_c; c++)
				for(int oc = 0; oc < filter_o_c; oc++)
					tmp_h_w[oc * filter_c + c] = *filter_h_w++;
		}
	}
	return tmp;
}

inline void rotate_180(float *filter, int filter_h, int filter_w, int filter_c, int filter_o_c)
{
	int h_size = filter_w * filter_c * filter_o_c;
	int w_size = filter_c * filter_o_c;
	for(int h = 0; h < filter_h / 2; h++)
		for(int w = 0; w < filter_w; w++)
		{
			float *filter_tmp1 = filter + h * h_size + w * w_size;
			float *filter_tmp2 = filter + (filter_h - 1 - h) * h_size + (filter_w - 1 - w) * w_size;
			for(int c = 0; c < w_size; c++)
				swap_f(filter_tmp1++, filter_tmp2++);
		}
	if(filter_h % 2)
	{
		int h = filter_h / 2;
		for(int w = 0; w < filter_w / 2; w++)
		{
			float *filter_tmp1 = filter + h * h_size + w * w_size;
			float *filter_tmp2 = filter + (filter_h - 1 - h) * h_size + (filter_w - 1 - w) * w_size;
			for(int c = 0; c < w_size; c++)
				swap_f(&*filter_tmp1++, &*filter_tmp2++);
		}
	}
}




int conv2d( float* input,	int input_batch,	int input_h,	int input_w,
			float* filter,	int filter_h,		int filter_w,	int filter_c,	int filter_o_c,
			float* output,	int output_h,		int output_w,
			int up,			int left, 			int need_to_rotate)
{
	if(need_to_rotate) // for gradient calculation
	{
		filter = swap_dim2_and_dim3( filter, filter_h, filter_w, filter_c, filter_o_c);
		swap_i(&filter_c, &filter_o_c);
		rotate_180(			filter, filter_h, filter_w, filter_c, filter_o_c);
	}

	int batch_size = input_h * input_w * filter_c;
	int batch_h_size = input_w * filter_c;
	int batch_h_w_size = filter_c;
	int output_batch_size = output_h * output_w * filter_o_c;

	int copy_size = filter_c * sizeof(float);

	int dim1 = output_h * output_w;
	int dim2 = filter_h * filter_w * filter_c;
	int dim3 = filter_o_c;

	float *ptr_input_batch = input;
	float *ptr_output_batch = output;

	if(mem_len < dim1 * dim2)
	{
		free(mem);
		mem = (float*)malloc((mem_len = dim1 * dim2) * sizeof(float));
	}

	float *img = mem; // temporary vector for calculation
	memset(img, 0, sizeof(float) * dim1 * dim2);

	for(int batch = 0; batch < input_batch; batch++, ptr_input_batch += batch_size, ptr_output_batch += output_batch_size) // for each patch
	{

		float *ptr_input_batch_h = ptr_input_batch;
		for(int _i_h = -up; _i_h < input_h - up; _i_h ++)
		{
			for(int _i_w = -left; _i_w < input_w - left; _i_w ++)
			{
				int i_h = _i_h + up, i_w = _i_w + left;
				int shift_size = (max(0, _i_w) - _i_w) * filter_c, move_size = filter_w * filter_c;
				float *ptr_input_batch_h_w_h2 = ptr_input_batch_h + (_i_h * input_w + _i_w) * filter_c + shift_size;
				float *ptr_img = img + (i_h * output_w + i_w) * filter_h * filter_w * filter_c + shift_size;

				int w_len = min(output_w, _i_w + filter_w) - max(0, _i_w);
				int copy_len = w_len * copy_size;

				for(int i_h2 = 0, i_h2_ = min(filter_h, output_h - _i_h); i_h2 < i_h2_; i_h2++,  ptr_input_batch_h_w_h2 += batch_h_size,
															ptr_img += move_size)
				{
					if(i_h2 + _i_h < 0) continue;
					memcpy(ptr_img, ptr_input_batch_h_w_h2, copy_len);
				}
			}
		}
		matmul(img, filter, ptr_output_batch, dim1, dim2, dim3, 0.0);
	}
	return 0;
}


int conv2d_grad(float* input,	int input_batch,	int input_h,	int input_w,
				float* grad,	int grad_h,			int grad_w,	
				float* output,	int output_h,		int output_w,	int output_c,	int output_o_c,
				int up,			int left)
{

	int batch_size = input_h * input_w * output_c;
	int batch_h_size = input_w * output_c;
	int batch_h_w_size = output_c;
	int grad_batch_size = grad_h * grad_w * output_o_c;
	int grad_batch_h_size = grad_w * output_o_c;
	int output_h_size = output_w * output_c * output_o_c;
	int output_h_w_size = output_c * output_o_c;

	int dim1 = output_h * output_w * output_c;
	int dim2 = grad_h * grad_w;
	int dim3 = output_o_c;

	float *ptr_input_batch = input;
	float *ptr_grad_batch = grad;

	if(mem_len < dim1 * dim2)
	{
		free(mem);
		mem = (float*)malloc((mem_len = dim1 * dim2) * sizeof(float));
	}

	float *img = mem; // temporary vector for calculation
	memset(img, 0, sizeof(float) * dim1 * dim2);

	for(int batch = 0; batch < input_batch; batch++, ptr_input_batch += batch_size, ptr_grad_batch += grad_batch_size) // for each patch
	{
		float *ptr_img = img;
		for(int _i_h = -up, _i_h_ = output_h - up; _i_h < _i_h_; _i_h ++)
		{
			int i_h = _i_h + up;
			for(int _i_w = -left, _i_w_ = output_w - left; _i_w < _i_w_; _i_w ++)
			{
				int i_w = _i_w + left;
				float *ptr_input_batch_h_w_h2 = ptr_input_batch + (_i_h * input_w + _i_w) * output_c;
				float *ptr_img_begin_pos = ptr_img + (i_h * output_w + i_w) * output_c * dim2;
				for(int i_h2 = max(0, -_i_h), i_h2_ = min(grad_h, input_h - _i_h); i_h2 < i_h2_; i_h2++)
				{
					float *ptr_img_begin_pos2 = ptr_img_begin_pos + i_h2 * grad_w + max(0, -_i_w);
					float *ptr_input_batch_h_w_h2_w2 = ptr_input_batch_h_w_h2 + i_h2 * input_w * output_c + max(0, -_i_w) * output_c;
					for(int i_w2 = max(0, -_i_w), i_w2_ = min(grad_w, input_w - _i_w); i_w2 < i_w2_; i_w2++)
					{
						float *tmp = ptr_img_begin_pos2;
						for(int c = 0; c < output_c; c++)
						{
							*tmp = ptr_input_batch_h_w_h2_w2[c];
							tmp += dim2;
						}
						++ptr_img_begin_pos2;
						ptr_input_batch_h_w_h2_w2 += output_c;
					}
				}
			}
		}
		matmul(img, ptr_grad_batch, output, dim1, dim2, dim3, batch > 0 ? 1.0 : 0.0);
	}
	return 0;
}


int maxpool( float* pos,
					float* input,	int input_h,		int input_w,
			 		float* output,	int output_batch,	int output_h,	int output_w,	int output_c,
			 		int stride_h, 	int stride_w,
			 		int up, 		int left)
{
	int input_batch_size = input_h * input_w * output_c;
	int input_batch_h_size = input_w * output_c;
	int output_batch_size = output_h * output_w * output_c;
	int output_batch_h_size = output_w * output_c;
	int idx = 0;
	memset(output, 254, sizeof(float) * output_batch * output_batch_size);
	for(int batch = 0; batch < output_batch; batch++)
	{
		float *ptr_input_batch = input + batch * input_batch_size;
		float *ptr_output_batch = output + batch * output_batch_size;
		float *ptr_pos_batch = pos + batch * output_batch_size;

		for(int _h = 0; _h < input_h; _h++)
		{
			for(int _w = 0; _w < input_w; _w++)
			{
				int h = _h + up, w = _w + left;

				++idx;

				float *ptr_input_cur = ptr_input_batch + _h * input_batch_h_size + _w * output_c;
				float *ptr_output_cur = ptr_output_batch + h / stride_h * output_batch_h_size + w / stride_w * output_c;
				float *ptr_pos_cur = ptr_pos_batch + h / stride_h * output_batch_h_size + w / stride_w * output_c;

				for(int c = 0; c < output_c; c++)
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

int maxpool_grad(float *pos, 	
						float* grad,	int grad_h,			int grad_w,
				 		float* output,	int output_batch,	int output_h,		int output_w,	int output_c,
		 				int stride_h, 	int stride_w, 		int up, 			int left)
{
	int grad_batch_size = grad_h * grad_w * output_c;
	int grad_batch_h_size = grad_w * output_c;
	int output_batch_size = output_h * output_w * output_c;
	int output_batch_h_size = output_w * output_c;
	int idx = 0;

	for(int batch = 0; batch < output_batch; batch++)
	{
		float *ptr_grad_batch = grad + batch * grad_batch_size;
		float *ptr_pos_batch = pos + batch * grad_batch_size;
		float *ptr_output_batch = output + batch * output_batch_size;

		for(int _h = 0; _h < output_h; _h++)
		{
			for(int _w = 0; _w < output_w; _w++)
			{
				int h = _h + up, w = _w + left;

				++idx;

				float *ptr_grad_cur = ptr_grad_batch + h / stride_h * grad_batch_h_size + w / stride_w * output_c;
				float *ptr_pos_cur = ptr_pos_batch + h / stride_h * grad_batch_h_size + w / stride_w * output_c;
				float *ptr_output_cur = ptr_output_batch + _h * output_batch_h_size + _w * output_c;

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

int sgn_zero_or_posi(float *input, float *grad, float *output, int len)
{
	for(int i = 0; i < len; i++, grad++) *output++ = *input++ > 0 ? *grad : 0;
	return 0;
}

int relu(float *input, float *output, int len)
{
	for(int i = 0; i < len; i++, ++input) *output++ = *input > 0 ? *input : 0;
	return 0;
}