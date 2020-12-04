#pragma once

#include <cstdio>
#include <algorithm>

namespace NN {
	class File {
	public:
		static bool is_little_endian() {
			short int number = 0x1;
			char* numPtr = (char*)&number;
			return (numPtr[0] == 1);
		}

		template<typename T>
		static T invert(T val) {
			unsigned char* p = reinterpret_cast<unsigned char*>(&val);
			int val_size = sizeof(T);
			for (int i = 0; i < (val_size >> 1); i++) {
				swap(p[i], p[val_size - 1 - i]);
			}

			T* new_val = reinterpret_cast<T *>(p);

			return *new_val;
		}

		File(FILE* file) {
			this->file = file;
		}

		void set_inverse(bool inverse) {
			this->inverse = inverse;
		}

		template<typename T>
		void save(T& var) {
			fwrite(&var, sizeof(T), 1, file);
		}

		template<typename T>
		void save_array(T* arr, int arr_size) {
			fwrite(arr, sizeof(T), arr_size, file);
		}

		template<typename T>
		void load(T& var) {
			fread(&var, sizeof(T), 1, file);
			if (inverse) var = invert(var);
		}

		template<typename T>
		void load_array(T* arr, int arr_size) {
			fread(arr, sizeof(T), arr_size, file);
			if (inverse) {
				for (int i = 0; i < arr_size; i++) {
					arr[i] = invert(arr[i]);
				}
			}
		}

		/*static void save(FILE* file, int& var);
		static void save(FILE* file, bool& var);
		static void save_float_array(FILE* file, float* arr, int arr_size);

		static void load(FILE* file, float& var, bool inverse);
		static void load(FILE* file, int& var, bool inverse);
		static void load(FILE* file, bool& var, bool inverse);
		static void load_float_array(FILE* file, float* arr, int arr_size, bool inverse);*/

	private:
		FILE* file;
		bool inverse = 0;
	};
}