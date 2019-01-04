#include <bits/stdc++.h>
using namespace std;

int main(){
	
	ifstream fin("raw_psr_list.txt");
	ofstream fout("psr_list.txt");
	
	string str1,str2,str3,str4,str5,str6,str7,str8;
	
	while(!fin.eof()){
		fin >> str1;
		fin >> str2;
		fin >> str3; //no use
		fin >> str4; //no use
		fin >> str5; //no use
		fin >> str6; //no use
		fin >> str7;
		fin >> str8; //no use
		int len = str7.length();
		bool good = false;
		for(int i=0;i<len;i++){
			if(str7[i]=='m'){
				good = true;
			}
		}
		fout << str1 << " " << str2 << endl;
		fout << good << endl;
	}
	
	fin.close();
	fout.close();
	
	cout << "Done!" << endl;
	
	return 0;
}
