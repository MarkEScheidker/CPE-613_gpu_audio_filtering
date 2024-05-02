#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <sys/time.h>
#include <stdlib.h>
#include "B_48000.h"

using namespace std;

struct header {
    unsigned int ChunkID;
    unsigned int ChunkSize;
    unsigned int Format;
    unsigned int Subchunk1ID;
    unsigned int Subchunk1Size;
    unsigned short AudioFormat;
    unsigned short NumChannels;
    unsigned int SampleRate;
    unsigned int ByteRate;
    unsigned short BlockAlign;
    unsigned short BitsPerSample;
    unsigned int Subchunk2ID;
    unsigned int Subchunk2Size;
};

int main() {

    string infile_name, outfile_name;     //declare file variables
    FILE *infile;
    FILE *outfile;
    short data;
    struct timeval start, end;
    
    infile_name = "white_noise.wav";
    outfile_name = "output.wav";

    infile = fopen(infile_name.c_str(), "rb");  //open in binary mode
    outfile = fopen(outfile_name.c_str(), "wb");
    
    //quit if files cannot be opened
    if (infile == NULL) {
      cout << "Error: Could not open input file." << endl;
      return 1;
    }
    if (outfile == NULL) {
      cout << "Error: Could not open output file." << endl;
      return 1;
    }
    
    //create header struct and fill with data from input file, checking for supported sample rate before continuing
    struct header my_header;
    fread(&my_header, sizeof(my_header),1, infile);

    if(my_header.NumChannels != 1){
        cout<<"only one channel supported"<<endl;
        return 0;
    }
    
    //write the entire header to the new file
    fwrite(&my_header, sizeof(my_header),1, outfile);
    
    //create an array for chanel with the size of the convolution array, filling it with zeros
    float data_buffer[BL];
    for(int i = 0; i<BL; i++){
        data_buffer[i] = 0.0;
    }
    
    //initialize variables to accumulate convolution of each sample
    float temp;
    
    //start the timer
    gettimeofday(&start, NULL);
    
    //iterate through every sample of the file
    for (int i = 0; i < my_header.Subchunk2Size / 2; i++){
        //shift all sample array items to the right by one
        for(int j = BL-1; j>0; j--){
            data_buffer[j] = data_buffer[j-1];
        }
        //read in new data from file
        fread(&data, sizeof(short), 1, infile);
        
        //recast data from int sample to double 
        data_buffer[0] = (float)(data); 
        
        //set accumulators to zero
        temp = 0.0;
        
        //calculate the value of one sample (L and R) by multiplying and accumulating the data buffer and the convolution array
        for(int j=0; j<BL; j++){
            temp += data_buffer[j] * B[j];
        }
        
        //convert accumulator back to int
        data = ((short)(temp));
        
        //write data into new file
        fwrite(&data, sizeof(short), 1, outfile);
    }
    
    //output compute time 
    gettimeofday(&end, NULL);  
    double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0; // Convert to milliseconds
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0; // Add microseconds as milliseconds
    cout<<"Program Runtime: "<<elapsed_time<<"ms"<<endl;  
}