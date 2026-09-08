#ifndef ENUMTYPES_H_
#define ENUMTYPES_H_


struct simControlStruct {
	enum class modelEnum {normal, edgeOnly, areaOnly, softBody, abnormal, hybrid, rounded, areaSquared} modelType;
	enum class neighborTypeEnum {cells, balls} neighborType = neighborTypeEnum::balls;
};

enum class minimizerEnum { GD, FIRE };

#endif /* ENUMTYPES_H_ */
