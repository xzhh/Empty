#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

using namespace std;


int main(int argc,char **argv){

	int i,j,count;
	int nbin=999999;
	double tmp;
	long double s[2][999999];
	for (i=0;i<999999;i++) s[1][i]=.0;
	if (argc<3){
cout<<"Usage: [binary] [spec_data_01  spec_data_02 .. ] output_file\n";
cout<<"       define bin by 'export N_BIN=XXX' to split one file";
return 0;
}
	if (getenv("N_BIN")!=NULL) nbin=atoi(getenv("N_BIN"));

	double num=argc-2.0;
	for (j=1;j<argc-1;j++){
		ifstream in(argv[j],ios::binary);
		i=0;
		while(in>>s[0][i%nbin]){
			in>>tmp;
			s[1][i%nbin]+=tmp;
			i++;
		}
		count=i;
		if (count<nbin) nbin=count;
	}
	ofstream out(argv[j],ios::binary);
	for (i=0;i<nbin;i++) out<<s[0][i]<<"\t"<<s[1][i]/num/double(count/nbin)<<endl;

	return 0;
}
