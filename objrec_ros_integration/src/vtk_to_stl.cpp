#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkSTLWriter.h>
#include <vtkDecimatePro.h>
 
int main(int argc, char *argv[])
{
  std::string inputFileName = argv[1];
  std::string outputFileName = argv[2];
 
  vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
  reader->SetFileName(inputFileName.c_str());
  reader->Update();



  vtkSmartPointer<vtkPolyData> input =
    vtkSmartPointer<vtkPolyData>::New();
  input->ShallowCopy(reader->GetOutput());


  vtkSmartPointer<vtkDecimatePro> decimate =
    vtkSmartPointer<vtkDecimatePro>::New();
#if VTK_MAJOR_VERSION <= 5
  decimate->SetInput(input);
#else
  decimate->SetInputData(input);
#endif
  //decimate->SetTargetReduction(.99); //99% reduction (if there was 100 triangles, now there will be 1)
  decimate->SetTargetReduction(.80); //10% reduction (if there was 100 triangles, now there will be 90)
  decimate->Update();
 
  vtkSmartPointer<vtkPolyData> decimated =
    vtkSmartPointer<vtkPolyData>::New();
  decimated->ShallowCopy(decimate->GetOutput());

  vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
  writer->SetFileName(outputFileName.c_str());
#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(decimated);
#else
  writer->SetInputData(decimated);
#endif  
  writer->SetFileTypeToBinary();
  writer->Update();
 
  return EXIT_SUCCESS;
}

