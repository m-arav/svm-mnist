#include<iostream>
#include <fstream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat read_mnist(char *);
Mat read_labels(char *);
int reverseInt(int);
vector<float> drawImg(vector< vector<int> >);   // change name to extract features

HOGDescriptor hog( Size(28,28), Size(14,14), Size(7,7), Size(7,7), 9); //  2 cells per block ; 7x7 = cell size
int n_images=10000;
// add feature number =  324

Mat read_labels(char* tr_lbl){
        
        Mat tr_labels(n_images, 1, CV_32SC1);
        ifstream labels(tr_lbl,ios::in | ios::binary);

        if (labels.is_open()){
                int magic_number,number_images;
                unsigned char lbl;

                labels.read((char*)&magic_number,sizeof(magic_number));
                magic_number= reverseInt(magic_number);
                labels.read((char*)&number_images,sizeof(number_images));
                number_images= reverseInt(number_images);

                for(int i=0;i<n_images;++i){
                    labels.read((char*)&lbl,sizeof(lbl));
                    tr_labels.at<int>(i,0)=(int)lbl;
                }
                //cout<<endl<<tr_labels<<endl;
		return tr_labels;
            }
            else cout<<"File I/O error"<<endl;
            return tr_labels;
}

vector<float> drawImg(vector< vector<int> > img){

        Mat my_mat(img.size(), img.size(), CV_8UC1);
        vector<float> ders;
        vector<Point> locs;

        for (size_t i = 0; i < img.size(); i++)
        {   
            for (size_t j = 0; j < img.size(); j++)
            {   
                my_mat.at<char>(i,j) = img[i][j];
            }   
        }
        hog.compute(my_mat,ders,Size(0,0),Size(0,0),locs);   
        return ders;
}

int reverseInt (int i){
    //reverse a int from high-endian to low-endian
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



Mat read_mnist(char* tr_img){

        
        Mat tr_data(n_images,324, CV_32FC1);
        
        ifstream data(tr_img,ios::in | ios::binary);
        if (data.is_open()){
                int magic_number,number_images,n_rows,n_cols;
                // Start reading Database
                data.read((char*)&magic_number,sizeof(magic_number));
                magic_number= reverseInt(magic_number);
                data.read((char*)&number_images,sizeof(number_images));
                number_images= reverseInt(number_images);
                data.read((char*)&n_rows,sizeof(n_rows));
                n_rows= reverseInt(n_rows);
                data.read((char*)&n_cols,sizeof(n_cols));
                n_cols= reverseInt(n_cols);
		
                cout<<"magic_number="<<magic_number<<endl;
                cout<<"number_images="<<number_images<<endl;
                vector< vector<int> > img(n_rows,vector<int>(n_cols));
		vector< vector<float> > features(n_images,vector<float>(324));
		
                /** Reading each images into img 2D vector**/
                for(int i=0; i<n_images; ++i){
                    for(int r=0; r<n_rows; ++r){
                        for(int c=0; c<n_cols; ++c){
                            unsigned char temp=0;
                            data.read((char*)&temp,sizeof(temp));
                            img[r][c]=(int)temp;                   
                        }
                    }       
                     features[i]=drawImg(img);   
                }
                for(int k=0;k<n_images;++k){
		        for(int j=0;j<324;++j){
			        tr_data.at<float>(k,j)=features[k][j];
		        }
	        }
                return tr_data;
        }
        else cout<<"File I/O error"<<endl;
        return tr_data;
}

int main(){
      

        Mat tr_data(n_images,324,CV_32FC1);
        Mat tr_label(n_images,1,CV_32SC1);
        Mat results(n_images,1, CV_32SC1);
        Mat result(1,1, CV_32SC1);
        
        char tr_img[]="train-images.idx3-ubyte";       
        char tr_lbl[]="train-labels.idx1-ubyte";
        char ts_img[]="t10k-images.idx3-ubyte";
        char ts_lbl[]="t10k-labels.idx1-ubyte";
        tr_data=read_mnist(tr_img);   
        tr_label=read_labels(tr_lbl);
  
        CvSVM SVM;
        CvSVMParams params;
        params.svm_type    = CvSVM::NU_SVC;
        params.kernel_type = CvSVM::POLY;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.degree        = CvSVM::POLY;
	params.gamma         = CvSVM::POLY;
	params.coef0         = CvSVM::POLY;
	params.nu            = 0.1;
	params.p             = CvSVM::EPS_SVR;
        SVM.train(tr_data, tr_label, Mat(), Mat(), params);
          
        tr_data=read_mnist(ts_img);   
        tr_label=read_labels(ts_lbl);
        SVM.predict(tr_data,results);

        int count=0;
        for(int i=0;i<n_images;++i){
                    if(tr_label.at<int>(i,0)==results.at<int>(i,0)){
                        //cout<<"stub";
                         ++count;
                    }
        }
        //float a=(float)count/n_images ;
        //cout<<"accuracy = "<<(float)a*100;
        Mat im,im_gray,blur,thresh,contourImage;
        vector<vector<Point> > contours;
        im = imread("photo_5.jpg", 1);
        cvtColor(im, im_gray,COLOR_BGR2GRAY);
        GaussianBlur(im_gray, blur, Size(5, 5), 0, 0);
   
        threshold(blur,thresh, 90, 255,THRESH_BINARY_INV);
        thresh.copyTo(contourImage);
        findContours(contourImage,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        for( unsigned int i = 0; i < contours.size(); i++ ){
              approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
              boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        }

        Mat roi,tmp;
        vector<float> ders;
        vector<Point>locs;
        Mat Hogfeat;
        for(unsigned int i = 0; i< contours.size(); i++ ){
                 rectangle( im, boundRect[i].tl(), boundRect[i].br()+Point(1,1), Scalar(0, 255, 0), 3);
                 roi = thresh(boundRect[i]);
                 resize(roi,tmp,Size(28, 28),INTER_AREA);
                 dilate(tmp,roi,Mat());
                 hog.compute(roi,ders,Size(0,0),Size(0,0),locs);
                 Hogfeat.create(ders.size(),1,CV_32FC1);
                 for(unsigned int i=0;i<ders.size();i++){
                   Hogfeat.at<float>(i,0)=ders.at(i);
                 }
                 imshow("test", im);
                 waitKey();
                 cout<<endl<<SVM.predict(Hogfeat,false)<<endl;
        }
        return 0;
}
