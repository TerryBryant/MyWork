#pragma once
/**
*  @file
*		vis_face_inpainter.h
*
*  @author
*		yangsong
*  @brief
*		Face inpainting based on dictionary
*
*  @details
*		For better recognition
*
*  @date
*		first bulid in Sep. 25th, 2017
*		first update in Oct. 20th, 2017, add "vis_init_facePoints" function
*
*  @version 1.0
*/
#ifndef VIS_FACE_INPAINTER_H_
#define VIS_FACE_INPAINTER_H_

#ifdef VIS_FACE_INPAINTER_EXPORTS
#define VIS_FACE_INPAINTER_API __declspec(dllexport)
#else
#define VIS_FACE_INPAINTER_API __declspec(dllimport)
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include "imageinterface.h"


#define DICT_IMAGE_HEIGHT 211 // the height of dict image
#define DICT_IMAGE_WIDTH 208  // the width of dict image
#define KEY_POINT_NUM 88   //the number of key points

using namespace std;
using namespace cv;

typedef void* MatHandle;//vector<Mat>*

// 错误代码
enum VisFaceInpainterErrorCode
{
	ErrorFaceInpainterOriImageFileLoad = 2,    //导入原始图像
	ErrorFaceInpainterMaskImageFileLoad,       //导入Mask
	ErrorFaceInpainterKeyPointsLoad,               //导入所有特征点
	ErrorFaceInpainterMatToBuffer,               //算法最后，将修复完的Mat格式图像转为buffer
};

// 单元修复区域尺寸的类型选择
enum class PatchSizeType
{
	AutoPatchSize = 0,   //程序自动判断尺寸
	BigPatchSize,
	SmallPatchSize
};

/************************************************************************/
/* 在区域内铺平均脸 */
/**
*  @fn vis_face_inpainter
*
*  @brief 铺点
*
*  @details
*		输入人脸区域，输出88个点
*
*  @param[in]	faceRect	输入人脸区域
*  @param[in out]    pt88   输出平均脸上的88个点

*  @return		返回值
*  @retval 1	表示操作成功，返回值为其它请参考VisFaceInpainterErrorCode
*
*/
/************************************************************************/
VIS_FACE_INPAINTER_API bool vis_init_facePoints(const dvRect & faceRect, const string& ref_model_filename, vector<dvPoint2D32f> & pt88);

/************************************************************************/
/* 导入字典及模型数据  */
/**
*  @fn vis_load_dictionary
*
*  @brief 导入数据
*
*  @details
*		输入相应的文件路径，输出导入数据的结果
*
*  @param[in]	ref_dict_filename		字典数据的文件路径
*  @param[in]	ref_model_filename		模型数据的文件路径
*  @param[in out]	dict		存储字典数据的变量
*  @param[in out]	ref_landmarks		存储模型数据的变量
*
*  @return		返回值
*  @retval true	表示操作成功，否则请检查文件路径
*
*/
/************************************************************************/
VIS_FACE_INPAINTER_API bool vis_load_dictionary(const string& ref_dict_filename, const string& ref_model_filename, MatHandle& dict, vector<dvPoint>& ref_landmarks);

/************************************************************************/
/* 回收字典及模型数据 */
/**
*  @fn vis_free_dictionary
*
*  @brief 回收数据
*
*  @details
*		输入存储字典及模型的变量，对其进行回收
*
*  @param[in]	dict		存储字典数据的变量
*  @param[in]	ref_landmarks		存储模型数据的变量
*
*  @return		无
*
*/
/************************************************************************/
VIS_FACE_INPAINTER_API void vis_free_dictionary(MatHandle &dict, vector<dvPoint> &ref_landmarks);

/************************************************************************/
/* 人脸修复操作 */
/**
*  @fn vis_face_inpainter
*
*  @brief 人脸修复
*
*  @details
*		输入人脸修复所需数据，输出最终的修复结果
*
*  @param[in]	pSrcImgOri		    输入原图像的buffer
*  @param[in]	pSrcImgMask		输入原图像Mask的buffer
*  @param[in]	ref_landmarks		输入模型数据
*  @param[in]	in_landmarks		    输入原图像所有特征点的坐标
*  @param[in]   dict	                        输入字典数据
*  @param[in]   patch_size	            输入单元修复区域尺寸
*  @param[in out]    pDstCdvImageInterface   输出修复后图像的buffer
*  @param[in out]	pTotal		                              输出总进度
*  @param[in out]	pCur		                              输出当前进度
*
*  @return		返回值
*  @retval 1	表示操作成功，返回值为其它请参考VisFaceInpainterErrorCode
*
*
*  修改记录：
*         Oct. 12th, 2017    输入参数增加patch_size，用户可自行指定单元修复区域的大小
*
*/
/************************************************************************/
VIS_FACE_INPAINTER_API int vis_face_inpainter(CdvImageInterface* pSrcImgOri, CdvImageInterface* pSrcImgMask, const vector<dvPoint>& ref_landmarks, const vector<dvPoint2D64f>& in_landmarks, const MatHandle& dict, const PatchSizeType& patch_size_type, CdvImageInterface* pDstCdvImageInterface, int* pTotal, int* pCur);





#endif // !VIS_FACE_INPAINTER_H_