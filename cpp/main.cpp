#include <iostream>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <fstream>
#include "itkMultiScaleTubularityMeasureImageFilter.hxx"
int main(){

  //typedefs for itk
  typedef int PixelType;
  const unsigned int DIMENSION = 3;
  typedef itk::Image<PixelType, DIMENSION> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  //create input and output file lists for file names
  //also create reader and writer for metaimage
  std::cout <<"Running OOF\n";
  std::ifstream file;
  file.open("./images.txt");
  std::string str;
  std::ifstream out_file;
  out_file.open("./output_images.txt");
  std::string out_str;
  //
  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  //loop over inputs, process and write to output
  while (std::getline(file,str)){
    std::cout << "Processing file: " << str << "\n";
    std::getline(out_file,out_str);
    std::cout << "outputting to: " << out_str << "\n";
  }
  return 0;
}
