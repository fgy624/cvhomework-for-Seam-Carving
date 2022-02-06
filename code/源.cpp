//��Ŀ���ƣ�ʵ��Seam Carving�㷨��ʵ��ͼ������
//���ߣ�fgy
//ʱ�䣺2022/1/18 20:58
//IDE��vs2019

# pragma warning (disable:4819)
#include <iostream>
#include <opencv2/opencv.hpp>
#include<string>
#include <vector>
#include<algorithm>
#include<stack>

using namespace std;
using namespace cv;

//ԭͼ����
int row_source;
int col_source;

//�任��ͼ����
int row_require;
int col_require;

//�洢����i��j�����ص����Ȩֵ��
float M[2010][2010];


//�洢����Сseam·��
int path[2010][2010];

//��ͼ,ԭʼͼ
Mat img = imread("C:\\Users\\admin\\Desktop\\test_10.jpg");
//����ɾ�����ͼ
Mat img_objectremove;



//����Ȩ��ģ��
//ÿһ�����ص�Ȩ��Ϊ�����ݶ��������ݶȵľ���ֵ֮��
//����������ÿ��������ݶȣ��õ���������
//�����������Ա�ľ���ֵ��Ӽ��ɵõ�����Ȩ�ؾ���
Mat calc_weight(Mat &img) {

	Mat temp(img.rows, img.cols, CV_8U);
	//��rgbͼ��Ϊ�Ҷ�ͼ��������ʱͼtemp��
	cvtColor(img, temp, COLOR_BGR2GRAY);
	cv::Mat gradiant_H(img.rows, img.cols, CV_32F, cv::Scalar(0)); //ˮƽ�ݶȾ���
	cv::Mat gradiant_V(img.rows, img.cols, CV_32F, cv::Scalar(0)); //��ֱ�ݶȾ���

	cv::Mat kernel_mine_H = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 1, -1, 0, 0, 0); //��ˮƽ�ݶ���ʹ�õľ���ˣ�����ʼֵ��
	cv::Mat kernel_mine_V = (cv::Mat_<float>(3, 3) << 0, 0, 0, 0, 1, 0, 0, -1, 0); //��ֱ�ݶ���ʹ�õľ���ˣ�����ʼֵ��

	cv::filter2D(temp, gradiant_H, gradiant_H.depth(), kernel_mine_H);
	cv::filter2D(temp, gradiant_V, gradiant_V.depth(), kernel_mine_V);

	cv::Mat gradMag_mat(img.rows, img.cols, CV_32F, cv::Scalar(0));
	cv::add(cv::abs(gradiant_H), cv::abs(gradiant_V), gradMag_mat);
	return gradMag_mat;
}

void dp_getWeakSeam(Mat &img) {//�õ���ֱ��������seam·��
	
	//cout << "img: " << img.type() << endl;
	int r = img.rows;
	int c = img.cols;

	//����ֵ����M����ĵ�һ�и������ر����Ȩ�أ�·���ڵ㸳�������ڵ�����
	for (int i = 0; i < c; i++) {
		M[0][i] = img.at<float>(0, i);
		path[0][i] = i;
	}

	//��̬�滮Ѱ�����Ȩֵ��
	//��̬�滮�Ĺ�ʽΪ��M[i,j]=e[i,j]+min(M[i-1,j-1], M[i-1][j], M[i-1][j+1])�����ǳ�min�е�������Ϊ��
	//���ϣ�up_l�����ϣ�up_m�����ϣ�up_r
	float up_l = 0.0;
	float up_m = 0.0;
	float up_r = 0.0;
	float add_weight = 0.0;
	for (int i = 1;  i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (j == 0) {
				up_m = M[i - 1][0];
				up_r = M[i - 1][1];
				add_weight = min(up_m, up_r);
				M[i][0] = add_weight + img.at<float>(i, 0);

				if (M[i - 1][0] == add_weight) {
					path[i][0] = 0;
				}
				else {
					path[i][0] = 1;
				}
			}
			else if (j == c - 1) {
				up_m = M[i - 1][j];
				up_l = M[i - 1][j - 1];
				add_weight = min(up_m, up_l);
				M[i][j] = add_weight + img.at<float>(i, j);

				if (M[i - 1][j] == add_weight) {
					path[i][j] = j;
				}
				else {
					path[i][j] = j - 1;
				}
			}
			else {
				up_l = M[i - 1][j - 1];
				up_r = M[i - 1][j + 1];
				up_m = M[i - 1][j];
				add_weight = min(up_l, min(up_m, up_r));

				if (M[i - 1][j - 1] == add_weight) {
					path[i][j] = j - 1;
				}
				else if (M[i - 1][j + 1] == add_weight) {
					path[i][j] = j + 1;
				}
				else {
					path[i][j] = j;
				}

				float test_data = img.at<float>(i, j);
				//cout << "test_data: " << test_data << endl;
				M[i][j] = test_data + add_weight;
				
			}
		}
	}
}

//������С���������ɾȥ��С����������seam
Mat changeTosmaller(Mat &img,Mat &temp_seam, int show_choose) {
	
	Mat img_clon = img.clone();

	int c = img.cols;
	int r = img.rows;
	int min_seam_loc = 0;
	float min_seamWeight = FLT_MAX;
	//�õ���Сseam��ֵ��λ��
	for (int i = 0; i < c; i++) {
		if (M[r - 1][i] < min_seamWeight) {
			min_seamWeight = M[r - 1][i];
			min_seam_loc = i;
		}	
	}

	temp_seam.at<int>(r - 1, 0) = min_seam_loc;
	int b = min_seam_loc;
	for (int i = r - 2; i >= 0; i--) {
		temp_seam.at<int>(i, 0) = path[i + 1][b];
		b = path[i + 1][b];
	}

	circle(img_clon, Point(min_seam_loc, r - 1), 1, Scalar(0, 0, 255));
	//�õ����ͼ��
	Mat out_img(r, c - 1, CV_8UC3);
	//cout <<"out_img: " << out_img.type() << endl;
	for (int j = 0; j < min_seam_loc; j++) {		
		out_img.at<Vec3b>(r - 1, j) = img.at<Vec3b>(r - 1, j);
	}

	for (int j = min_seam_loc + 1; j < c; j++) {
		out_img.at<Vec3b>(r - 1, j - 1) = img.at<Vec3b>(r - 1, j);
	}

	for (int i = r - 2; i >= 0; i--) {
		for (int j = 0; j < path[i + 1][min_seam_loc]; j++) {
			out_img.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}

		for (int j = path[i + 1][min_seam_loc] + 1; j < c; j++) {
			out_img.at<Vec3b>(i, j - 1) = img.at<Vec3b>(i, j);
		}

		min_seam_loc = path[i + 1][min_seam_loc];
		circle(img_clon, Point(path[i + 1][min_seam_loc], i), 1, Scalar(0, 0, 255));
	}

	if (show_choose == 1) {
		imshow("��С��", img_clon);
		waitKey(50);
	}

	if (show_choose == 2) {
		Mat img_clon_rotate;
		transpose(img_clon, img_clon_rotate);
		flip(img_clon_rotate, img_clon_rotate, 0);
		imshow("��С��", img_clon_rotate);
		waitKey(50);
	}

	return out_img;
}

//�Ŵ�
Mat changeTolarger(Mat& img, Mat& seams,int adj_num,int show_choose) {

	Mat out_img(img.rows, img.cols + 1, CV_8UC3);
	//cout << "adj_num: " << adj_num << endl;
	Mat out_img_clo = img.clone();

	for (int i = 0; i < img.rows; i++) {
		
		int k = seams.at<int>(i, 0) + adj_num;
	
		int j = 0;
		for (j = 0; j < k; j++) {
			out_img.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
		}
		if (j == img.cols) {
			continue;
		}
		else if (j == img.cols - 1) {
			out_img.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[0]) / 2;
			out_img.at<Vec3b>(i, j)[1] = (img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j )[1]) / 2;
			out_img.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i, j)[2] + img.at<Vec3b>(i, j )[2]) / 2;
		}
		else {
			out_img.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j + 1)[0]) / 2;
			out_img.at<Vec3b>(i, j)[1] = (img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j + 1)[1]) / 2;
			out_img.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i, j)[2] + img.at<Vec3b>(i, j + 1)[2]) / 2;

			for (int jj = j + 1; jj < img.cols + 1; jj++) {
				out_img.at<Vec3b>(i, jj) = img.at<Vec3b>(i, jj - 1);
			}

		}

		circle(out_img_clo, Point(k, i), 1, Scalar(0, 0, 255));
		
	}

	if (show_choose == 1) {
		imshow("�Ŵ���", out_img_clo);
		waitKey(50);
		/*imshow("out", out_img);
		waitKey(50);*/
	}
	
	if (show_choose == 2) {
		Mat out_img_rotate;
		transpose(out_img_clo, out_img_rotate);
		flip(out_img_rotate, out_img_rotate, 0);
		imshow("�Ŵ���", out_img_rotate);
		waitKey(50);
	}
	return out_img;
}

float getMinM(Mat img) {
	int r = img.rows;
	int c = img.cols;

	int minseam_index_r = 0;
	float minseam_r = FLT_MAX;
	for (int i = 0; i < c; i++) {
		if (M[r - 1][i] < minseam_r) {
			minseam_index_r = i;
			minseam_r = M[r - 1][i];
		}
	}
	
	return minseam_r;
}

int getPixelnum(Mat mask) {
	int counter = 0;
	int r = mask.rows;
	int c = mask.cols;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (mask.at<Vec3b>(i, j)[0] == 255 && mask.at<Vec3b>(i, j)[1] == 255 && mask.at<Vec3b>(i, j)[2] == 255) {
				counter++;
			}
		}
	}

	return counter;
}

Mat objectRemove(Mat img_obj, Mat mask) {

	Mat out_img = img_obj;

	int pnum = getPixelnum(mask);
	vector<Mat> seams;//�洢seams
	while (pnum != 0) {

		Mat energy_mat(out_img.rows, out_img.cols, CV_32F);

		energy_mat = calc_weight(out_img);
		for (int i = 0; i < energy_mat.rows; i++) {
			for (int j = 0; j < energy_mat.cols; j++) {
				if (mask.at<Vec3b>(i, j)[0] == 255 && mask.at<Vec3b>(i, j)[1] == 255 && mask.at<Vec3b>(i, j)[2] == 255) {
					energy_mat.at<float>(i, j) = -1000.0;
				}
			}
		}
		dp_getWeakSeam(energy_mat);
		Mat temp_seam(img_obj.rows, 1, CV_32S);//��ǰ��Сseam
		out_img = changeTosmaller(out_img, temp_seam, 1);
		mask = changeTosmaller(mask, temp_seam, 3);

		pnum = getPixelnum(mask);
	}
	return out_img;
}


vector<Point>trace_mask;
Point start;
Mat img_objrv_clo;

void on_mouse(int event, int x, int y, int flags, void* ss) {
	Point p1, p2;
	
	if (event == CV_EVENT_LBUTTONDOWN) {//������
		cout << "1" << endl;
		start.x = x;
		start.y = y;
		circle(img_objectremove, start, 1, Scalar(0, 0, 255));
		trace_mask.push_back(start);
		imshow("��ͼ", img_objectremove);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) {//�����ס����
		p1.x = x;
		p1.y = y;

		cout << "2" << endl;
		//������
		line(img_objectremove, p1, trace_mask.back(), Scalar(0, 0, 255));
		circle(img_objectremove, start, 1, Scalar(0, 0, 255));
		trace_mask.push_back(p1);
		imshow("��ͼ", img_objectremove);
	}
	else if (event == CV_EVENT_LBUTTONUP) {//�ɿ����
		cout << "3" << endl;
		p1.x = x;
		p1.y = y;
		
		line(img_objectremove, p1, start, Scalar(0, 0, 255));
		circle(img_objectremove, start, 1, Scalar(0, 0, 255));
		trace_mask.push_back(p1);

		const Point* ppt[1] = { &trace_mask[0] };

		int npt[] = { trace_mask.size() };

		Mat mask = img_objectremove.clone();
		mask.setTo(Scalar(0, 0, 0));
		fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255));

		imshow("mask", mask);

		Mat out_img = objectRemove(img_objrv_clo, mask);

		imshow("���ͼ", out_img);
	}
}


int main() {
	
	cout << "������" << endl;
	cout << "1 ������С" << endl;
	cout << "2 �߶���С" << endl;
	cout << "3 ����ά���ϵ���С" << endl;
	cout << "4 ����Ŵ�" << endl;
	cout << "5 ����Ŵ�" << endl;
	cout << "6 Ŀ�����" << endl;
	cout << endl;

	row_source = img.rows;
	col_source = img.cols;
	cout << "ԭͼ�ĳ���Ϊ��" << col_source << endl;
	cout << "ԭͼ�ĸ߶�Ϊ��" << row_source << endl;
	cout << endl;

	int state_choose = 0;
	cout << "��ѡ����ͼ���ģʽ(����1-6)��" << endl;
	cin >> state_choose;
	cout << endl;

	//������С
	if (state_choose == 1) {
		cout << "�����������õ���ͼ�ĳ���" << endl;
		
		cin >> col_require;
		cout << endl;
		if (col_require > col_source) {
			cout << "���벻����" << endl;
			exit(0);
		}

		Mat temp = img.clone();
		vector<Mat> seams;//�洢seams
		Mat temp_seam(temp.rows, 1, CV_32S);//��ǰ��Сseam
		for (int i = col_require; i < col_source; i++) {

			Mat energy_img = calc_weight(temp);
			dp_getWeakSeam(energy_img);
			temp = changeTosmaller(temp, temp_seam, 1);
			seams.push_back(temp_seam);

		}

		imshow("���ͼ", temp);
		cout << "��ͼ��ĳ�Ϊ�� " << temp.cols << endl;
		cout << "��ͼ��ĸ�Ϊ�� " << temp.rows << endl;
	}

	//������С
	if (state_choose == 2) {
		cout << "�����������õ���ͼ�ĸߣ�" << endl;	
		cin >> row_require;
		cout << endl;

		if (row_require > row_source) {
			cout << "���벻����" << endl;
			exit(0);
		}

		Mat temp = img.clone();
		//������˳ʱ����תԭͼ���ٵ��ú�����С����
		Mat temp_rotate;

		transpose(temp, temp_rotate);
		flip(temp_rotate, temp_rotate, 1);

		vector<Mat> seams;//�洢seams
		Mat temp_seam(temp_rotate.rows, 1, CV_32S);//��ǰ��Сseam

		for (int i = row_require; i < row_source; i++) {

			Mat energy_img = calc_weight(temp_rotate);
			dp_getWeakSeam(energy_img);
			temp_rotate = changeTosmaller(temp_rotate, temp_seam, 2);
			seams.push_back(temp_seam);
		}

		transpose(temp_rotate, temp_rotate);
		flip(temp_rotate, temp_rotate, 0);
		imshow("���ͼ", temp_rotate);
		cout << "��ͼ��ĳ�Ϊ�� " << temp_rotate.cols << endl;
		cout << "��ͼ��ĸ�Ϊ�� " << temp_rotate.rows << endl;
	}

	//��ά����С
	if (state_choose == 3) {
		cout << "�����������õ�ͼ�ĳ�" << endl;
		cin >> col_require;
		cout << endl;

		cout << "�����������õ���ͼ�ĸ�" << endl;
		cin >> row_require;
		cout << endl;
		if (col_require > col_source || row_require > row_source) {
			cout << "���벻����" << endl;
			exit(0);
		}

		Mat temp = img.clone();
		vector<Mat> seams;//�洢seams
		Mat temp_seam(temp.rows, 1, CV_32S);//��ǰ��Сseam

		int change_total_num = (row_source - row_require) + (col_source - col_require);

		int counter_c = 0;
		int counter_r = 0;

		for (int i = 0; i < change_total_num; i++) {

			Mat temp_rotate;
			transpose(temp, temp_rotate);
			flip(temp_rotate, temp_rotate, 1);

			Mat energy_img_r = calc_weight(temp);
			Mat energy_img_c = calc_weight(temp_rotate);

			dp_getWeakSeam(energy_img_r);
			float minseam_c = getMinM(temp);

			dp_getWeakSeam(energy_img_c);
			float minseam_r = getMinM(temp_rotate);

			if (minseam_c < minseam_r) {
				if (counter_c < (col_source - col_require)) {
					Mat energy_img_r = calc_weight(temp);
					dp_getWeakSeam(energy_img_r);
					temp = changeTosmaller(temp, temp_seam, 1);
					counter_c++;
				}
				else {
					vector<Mat> seams;//�洢seams
					Mat temp_seam(temp_rotate.rows, 1, CV_32S);//��ǰ��Сseam
					temp_rotate = changeTosmaller(temp_rotate, temp_seam, 2);
					transpose(temp_rotate, temp);
					flip(temp, temp, 0);
					counter_r++;
				}
			}
			else {
				if (counter_r < (row_source - row_require)) {
					vector<Mat> seams;//�洢seams
					Mat temp_seam(temp_rotate.rows, 1, CV_32S);//��ǰ��Сseam
					temp_rotate = changeTosmaller(temp_rotate, temp_seam, 2);
					transpose(temp_rotate, temp);
					flip(temp, temp, 0);
					counter_r++;
				} 
				else {
					Mat energy_img_c = calc_weight(temp_rotate);
					dp_getWeakSeam(energy_img_r);
					temp = changeTosmaller(temp, temp_seam, 1);
					counter_c++;
				}
			}
		}
		cout << "temp:" << temp.rows <<" " << temp.cols << endl;
		imshow("���ͼ", temp);

		cout << "��ͼ��ĳ�Ϊ�� " << temp.cols << endl;
		cout << "��ͼ��ĸ�Ϊ�� " << temp.rows << endl;
	}
	
	//����Ŵ�
	if (state_choose == 4) {
		cout << "�����������õ���ͼ�ĳ���" << endl;
		cin >> col_require;
		cout << endl;

		if (col_require < col_source) {
			cout << "���벻����" << endl;
			exit(0);
		}

		Mat temp = img.clone();
		vector<Mat> seams;//�洢seams

		Mat temp_clo = temp.clone();
		for (int i = col_source; i < col_require; i++) {
			Mat e_img = calc_weight(temp);
			dp_getWeakSeam(e_img);
			Mat temp_seam(temp.rows, 1, CV_32S, Scalar(0, 0, 255));
			temp = changeTosmaller(temp, temp_seam, 3);
			seams.push_back(temp_seam); 
		}

		Mat adj_arr(col_require - col_source, 1, CV_32S, Scalar(0));
		for (int i = 0; i < col_require - col_source; i++) {
			for (int j = 0; j < col_require - col_source; j++) {
				if (seams[i].at<int>(temp.rows - 1, 0) < seams[j].at<int>(temp.rows - 1, 0)) {
					continue;
				}
				else {
					adj_arr.at<int>(i,0) += 1;
				}
			}
		}

		for (int i = col_source; i < col_require; i++) {
			int adj_num = adj_arr.at<int>(col_require - i - 1, 0);
			temp_clo = changeTolarger(temp_clo, seams[col_require - i - 1], adj_num, 1);
		}

		imshow("���ͼ", temp_clo);
		cout << "��ͼ��ĳ�Ϊ�� " << temp_clo.cols << endl;
		cout << "��ͼ��ĸ�Ϊ�� " << temp_clo.rows << endl;
	}

	//����Ŵ�
	if (state_choose == 5) {
		cout << "�����������õ���ͼ�ĸߣ�" << endl;
		cout << endl;

		cin >> row_require;
		if (row_require < row_source) {
			cout << "���벻����" << endl;
			exit(0);
		}

		Mat temp = img.clone();

		//������˳ʱ����תԭͼ���ٵ��ú���Ŵ󼴿�
		Mat temp_rotate;
		transpose(temp, temp_rotate);
		flip(temp_rotate, temp_rotate, 1);

		Mat temp_clo = temp_rotate.clone();
		vector<Mat> seams;//�洢seams
		Mat temp_seam(temp_rotate.rows, 1, CV_32S);//��ǰ��Сseam

		for (int i = row_source; i < row_require; i++) {
			Mat e_img = calc_weight(temp_rotate);
			dp_getWeakSeam(e_img);
			Mat temp_seam(temp_rotate.rows, 1, CV_32S, Scalar(0, 0, 255));
			temp_rotate = changeTosmaller(temp_rotate, temp_seam, 3);
			seams.push_back(temp_seam);
		}

		Mat adj_arr(row_require - row_source, 1, CV_32S, Scalar(0));
		for (int i = 0; i < row_require - row_source; i++) {
			for (int j = 0; j < row_require - row_source; j++) {
				if (seams[i].at<int>(temp_rotate.rows - 1, 0) < seams[j].at<int>(temp_rotate.rows - 1, 0)) {
					continue;
				}
				else {
					adj_arr.at<int>(i, 0) += 1;
				}
			}
		}

		for (int i = row_source; i < row_require; i++) {
			int adj_num = adj_arr.at<int>(row_require - i - 1, 0);
			temp_clo = changeTolarger(temp_clo, seams[row_require - i - 1], adj_num, 2);
		}

		transpose(temp_clo, temp_clo);
		flip(temp_clo, temp_clo, 0);
		imshow("���ͼ", temp_clo);
		cout << "��ͼ��ĳ�Ϊ�� " << temp_clo.cols << endl;
		cout << "��ͼ��ĸ�Ϊ�� " << temp_clo.rows << endl;
	}
	
	//Ŀ�����
	if (state_choose == 6) {

		img_objectremove = img.clone();
		img_objrv_clo = img.clone();
		namedWindow("��ͼ");
		imshow("��ͼ", img_objectremove);
	
		//ԭͼ�ϻ���ѡ��ɾ������
		setMouseCallback("��ͼ", on_mouse, 0);	

	}
	imshow("ԭͼ", img);
	waitKey(0);
	
}
