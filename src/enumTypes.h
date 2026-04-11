#ifndef ENUMTYPES_H_
#define ENUMTYPES_H_


struct simControlStruct {
	enum class modelEnum {normal, edgeOnly, areaOnly, softBody, abnormal} modelType;
};

#endif /* ENUMTYPES_H_ */
