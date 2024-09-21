
#include "miniply.h"
#include <iostream>
#include <fstream>
#include "Mesh.h"
#include "MeshPLY.h"

bool ReadPLY(miniply::PLYReader& reader, Mesh* pMesh)
{
    uint32_t faceIdxs[3];
    miniply::PLYElement* faceElem = reader.get_element(reader.find_element(miniply::kPLYFaceElement));
    if (faceElem != nullptr) {
        faceElem->convert_list_to_fixed_size(faceElem->find_property("vertex_indices"), 3, faceIdxs);
    }
    pMesh->clear();

    uint32_t indexes[3];
    bool gotVerts = false, gotFaces = false;

    while (reader.has_element() && (!gotVerts || !gotFaces)) {
        if (reader.element_is(miniply::kPLYVertexElement) && reader.load_element() && reader.find_pos(indexes)) {
            pMesh->mV.resize(reader.num_rows());
            reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float, pMesh->VData());

            if (reader.find_normal(indexes)) {
                // Also get normals
                pMesh->mN.resize(pMesh->nV());
                reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float, pMesh->NData());
            }

            if (reader.find_color(indexes)) {
                // Also get normals
                pMesh->mC.resize(pMesh->nV());
                C3* pColor3 = new C3[pMesh->nV()];
                reader.extract_properties(indexes, 3, miniply::PLYPropertyType::UChar, pColor3);
                for (int i = 0; i < pMesh->nC(); i++)
                    pMesh->mC[i] = C4{ 0x00, pColor3[i].r, pColor3[i].g, pColor3[i].b };
                delete[] pColor3;
            }
            gotVerts = true;
        }
        else if (!gotFaces && reader.element_is(miniply::kPLYFaceElement) && reader.load_element()) {
                pMesh->mF.resize(reader.num_rows());
                reader.extract_properties(faceIdxs, 3, miniply::PLYPropertyType::Int, pMesh->FData());
                gotFaces = true;
        }
        if (gotVerts && gotFaces) {
            break;
        }
        reader.next_element();
    }
    return true;
}



bool ReadPLY(std::string pPath, Mesh* pMesh)
{
    miniply::PLYReader reader(pPath.c_str());
    if (!reader.valid()) return false;

    return ReadPLY(reader, pMesh);
}

/////////////////////////////////////////////////////////////////////////////

bool WritePLY(std::ofstream& pFile, Mesh* pMesh){
    if (pMesh == nullptr) return false;
    if (!pFile.is_open()) return false;

    bool normalAvailable = pMesh->nN();
    bool vertexColorAvailable = pMesh->nC();

    //header ////////////////////////////////////////////////
    pFile << "ply\n";
    pFile << "format binary_little_endian 1.0\n";
    pFile << "element vertex " << pMesh->nV() << "\n";
    pFile << "property float x\n";
    pFile << "property float y\n";
    pFile << "property float z\n";

    if (normalAvailable){
        pFile << "property float nx\n";
        pFile << "property float ny\n";
        pFile << "property float nz\n";
    }

    if (vertexColorAvailable){
            pFile << "property uchar red\n";
            pFile << "property uchar green\n";
            pFile << "property uchar blue\n";
    }

    if (pMesh->nF())
    {
        pFile << "element face " << pMesh->nF() << "\n";
        pFile << "property list uchar uint vertex_indices\n";
    }
    pFile << "end_header\n";

    V3* pNormal = pMesh->NData();
    C4* pVColor = pMesh->CData();
    I3* pFaces  = pMesh->FData();
    C3 c3;

    for (int i = 0; i < pMesh->nV(); i++) {
        pFile.write((char*)(&pMesh->mV[i]), sizeof(pMesh->mV[i]));

        if (normalAvailable)
            pFile.write((char*)(&pNormal[i]), sizeof(pNormal[i]));

        if (vertexColorAvailable)
        {
            c3 = {pVColor[i].r, pVColor[i].g, pVColor[i].b};
            pFile.write((char*)(&c3), sizeof(c3));
        }
    }

    const char array_count = 3;
    for (int i = 0; i < pMesh->nF(); i++) {
        pFile.write((char*)&(array_count), sizeof(array_count));
        pFile.write((char*)(&pFaces[i]), sizeof(pFaces[i]));
    }

    return true;

}

bool WritePLY(const std::string& pPath, Mesh* pMesh)
{
    if (pMesh == nullptr) return false;
    std::ofstream outfile(pPath, WRITE_M);
    return WritePLY(outfile, pMesh);
}
