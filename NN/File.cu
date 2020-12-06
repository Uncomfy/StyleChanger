#include "File.h"

using namespace std;

namespace NN {
	/*bool File::is_little_endian() {
		short int number = 0x1;
		char* numPtr = (char*)&number;
		return (numPtr[0] == 1);
	}

	void File::save(FILE* file, float& var) {
		fwrite(&var, sizeof(float), 1, file);
	}

	void File::save(FILE* file, int& var) {
		fwrite(&var, sizeof(int), 1, file);
	}

	void File::save(FILE* file, bool& var) {
		fwrite(&var, sizeof(bool), 1, file);
	}

	void File::save_float_array(FILE* file, float* arr, int arr_size) {
		fwrite(arr, sizeof(float), arr_size, file);
	}

	void File::load(FILE* file, float& var, bool inverse) {
		fread(&var, sizeof(float), 1, file);
		if (inverse) var = invert(var);
	}

	void File::load(FILE* file, int& var, bool inverse) {
		fread(&var, sizeof(int), 1, file);
		if (inverse) var = invert(var);
	}

	void File::load(FILE* file, bool& var, bool inverse) {
		fread(&var, sizeof(bool), 1, file);
		if (inverse) var = invert(var);
	}

	void File::load_float_array(FILE* file, float* arr, int arr_size, bool inverse) {
		fread(arr, sizeof(float), arr_size, file);
		if (inverse) {
			for (int i = 0; i < arr_size; i++) {
				arr[i] = invert(arr[i]);
			}
		}
	}*/
}