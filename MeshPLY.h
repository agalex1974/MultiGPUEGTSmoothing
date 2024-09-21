#ifndef MESHPLY_H
#define MESHPLY_H

#include "Mesh.h"
static const std::ios_base::openmode WRITE_M = std::ios::binary | std::ios::out | std::ios::trunc;

// Internal Read Write Files

struct Mesh;

bool ReadPLY(std::string pPath, Mesh* pMesh);

bool WritePLY(const std::string& pPath, Mesh* pMesh);
bool WritePLY(std::ostream* pFile, Mesh* pMesh);

#endif