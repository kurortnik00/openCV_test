#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/types_c.h>

using namespace cv;
using namespace std;



static struct Counter
{
public:
	void incrementIN() 
	{
		++countIN;
		cout << "One more person has come, now " << countIN << " persons in.\n";
	}
	void incrementOUT() 
	{
		++countOUT;
		cout << "One more person has gone, now " << countOUT << " persons out.\n";
	}

	void print()
	{
		cout << "persons IN: " << countIN << "\n" << "pensons OUT: " << countOUT << "\n";
	}
	int getCountIN() { return countIN; }
	int getCounOUT() { return countOUT; }

private:
	int countIN = 0;
	int countOUT = 0;
};
Counter counter;

struct Person
{
public:
	Person(Rect rect)
		:rect_(rect), actualCounter(0)
	{
		center_of_rect = (rect_.br() + rect_.tl()) * 0.5;
		startPoint = center_of_rect;
		cout << "new Pesrson at "<< startPoint.x << "," << startPoint.y << "\n";
	}
	~Person()
	{
		int value = startPoint.x - center_of_rect.x;
		if (abs(value) > 50)
		{
			if (value > 0) counter.incrementOUT();
			if (value < 0) counter.incrementIN();
		}
		counter.print();
	}
	Point getCenter() { return center_of_rect; }
	Point getstartPoint() { return startPoint; }
	Rect getRect() { return rect_; }
	void setRect(Rect rect) 
	{ 
		rect_ = rect; 
		center_of_rect = (rect_.br() + rect_.tl()) * 0.5;
		actualCounter = 0;
	}
	void  incrementActualCounter()
	{
		++actualCounter;
	}
	int getActualCounter() { return actualCounter; }


private:
	Rect rect_;
	Point center_of_rect, startPoint;
	int actualCounter;
};


bool operator > (Rect const &r1, Rect const &r2)
{
	return r1.area() > r2.area();
}

void contoursProcedure(Mat& frame, vector<vector<Point>>& contours, vector<Person*>& persons);
void printFPS(Mat& frame, VideoCapture& capture);
void printFPS(Mat& frame);

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: " << argv[0] << " ImageToLoadAndDisplay" << endl;
		return -1;
	}
	VideoCapture capture(samples::findFile(argv[1]));
	if (!capture.isOpened()) 
	{
		//error in opening the video input
		cerr << "Unable to open: " << argv[1] << endl;
		return 0;
	}

	Mat frame, fgMask;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	int CV_RETR_TREE = 3;
	int CV_CHAIN_APPROX_SIMPLE = 1;
	Ptr<BackgroundSubtractor> pBackSub;
	pBackSub = createBackgroundSubtractorKNN(500, 900.0);

	int morph_size = 5;
	int morph_operator = 2;
	int morph_elem = 0;

	vector<Person*> persons;

	while (true) {
		
		capture >> frame;
		if (frame.empty())
			break;
		pBackSub->apply(frame, fgMask);

		threshold(fgMask, fgMask, 200, 255, 0);
		Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
		morphologyEx(fgMask, fgMask, morph_operator, element);
		GaussianBlur(fgMask, fgMask, Size(15, 15), 0, 0);
		findContours(fgMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		contoursProcedure(frame, contours, persons);
		//cv::imshow("FG Mask", fgMask);

		printFPS(frame);
		printFPS(frame, capture);
		cv::imshow("Source", frame);
		//get the input from the keyboard
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;

	}
	return 0;
}

void contoursProcedure(Mat& frame, vector<vector<Point>>& contours, vector<Person*>& persons)
{
	for (auto contur : contours)
	{
		if ((contourArea(contur) > 8500) && (contourArea(contur) < 150000)) ///"(contourArea(contur) < 150000)" need to avoid errors in first frames (probably whole image iterated like one boundary box)
		{
			Rect rect = boundingRect(contur);
			if (persons.empty()) persons.push_back(new Person(rect));
			else
			{
				bool isNewPerson = true;
				for (auto p : persons)
				{
					if (rect.contains(p->getCenter()))
					{
						p->setRect(rect);
						isNewPerson = false;
					}
				}
				if (isNewPerson)
				{
					persons.push_back(new Person(rect));
				}
			}
			cv::rectangle(frame, rect, (255, 0, 0), 2);
		}
	}
	for (auto p : persons)
	{
		p->incrementActualCounter();
	}
	if (!persons.empty())
	{
		if (persons[0]->getActualCounter() > 10)
		{
			delete persons[0];
			persons.erase(persons.begin());
		}
	}
}

void printFPS(Mat& frame, VideoCapture& capture)
{
	//get the frame number and write it on the current frame
	cv::rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
		cv::Scalar(255, 255, 255), -1);
	stringstream ss;
	ss << capture.get(CAP_PROP_POS_FRAMES);
	string frameNumberString = ss.str();
	cv::putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
		FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	//show the current frame and the fg masks
}


void printFPS(Mat& frame)
{
	//get the frame number and write it on the current frame
	cv::rectangle(frame, cv::Point(150, 320), cv::Point(400, 350),
		cv::Scalar(255, 255, 255), -1);
	stringstream ss;
	ss << counter.getCounOUT() << " <-OUT             "<< "IN -> " << counter.getCountIN();
	string frameNumberString = ss.str();
	cv::putText(frame, frameNumberString.c_str(), cv::Point(150, 335),
		FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	//show the current frame and the fg masks
}