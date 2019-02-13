//
// Created by mihai on 17.12.2018.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>

/*********** declararea tipurilor de date folosite si macro-urilor ************/

typedef enum {
    grayscale = 5,
    color = 6,
} TypeImage;

#pragma pack(1)

/**
 * Structura unui pixel "colorat"
 */
typedef struct {
    unsigned char r; /** culoarea rosu */
    unsigned char g; /** culoarea verde */
    unsigned char b; /** culoarea albastru */
} PixelColor;

/**
 * Structura unui pixel "gri"
 */
typedef struct {
    unsigned char pix;
} PixelGray;

/**
 * Structura unei imagini pnm sau pgm
 */
typedef struct {
    unsigned int width;  /** latimea imaginei */
    unsigned int height; /** inaltinea imaginei */
    void *matrix;        /** matricea de pixeli */
    TypeImage typeImage; /** tipul de imagine citita */
} Image;

/**
 * Structura care retine infomatile necesare unui program
 */
typedef struct {
    Image img;
    int start;
    int end;
    int id;
    int nrProcesses;
} Info;

#pragma pack()

#define R r
#define B b
#define G g
#define PIX pix

/**
 * macro-ul utilizat pentru calcularea valorii pe un canal de culoare
 * */
#define CALC_PIXEL(matrixColor, i_start, j_start, GaussianKernel, member) \
    (GaussianKernel[0] * matrixColor[i_start - 1][j_start - 1].member +   \
     GaussianKernel[1] * matrixColor[i_start - 1][j_start].member +       \
     GaussianKernel[2] * matrixColor[i_start - 1][j_start + 1].member +   \
     GaussianKernel[3] * matrixColor[i_start][j_start - 1].member +       \
     GaussianKernel[4] * matrixColor[i_start][j_start].member +           \
     GaussianKernel[5] * matrixColor[i_start][j_start + 1].member +       \
     GaussianKernel[6] * matrixColor[i_start + 1][j_start - 1].member +   \
     GaussianKernel[7] * matrixColor[i_start + 1][j_start].member +       \
     GaussianKernel[8] * matrixColor[i_start + 1][j_start + 1].member)

#define TYPEGRAY        PixelGray
#define TYPECOLOR       PixelColor

/**
 * macro-ul folosit pentru trimiterea/primirea informatilor necesare unui
 * proces, de la celelalte procese
 */
#define SHARE(image, infoProcess, len, send_down, send_up, TYPE)            \
    TYPE **matrix = (TYPE **) image->matrix;                                \
                                                                            \
    memcpy(send_up, matrix[image->height - 2], (size_t) len);               \
    memcpy(send_down, matrix[1], (size_t) len);                             \
                                                                            \
    /** up */                                                               \
    if (infoProcess.id == ROOT) {                                           \
        MPI_Send(send_up, len, MPI_CHAR, ROOT + 1, 0, MPI_COMM_WORLD);      \
    } else if (infoProcess.id == infoProcess.nrProcesses - 1) {             \
        MPI_Recv(matrix[0], len, MPI_CHAR, infoProcess.id - 1, 0,           \
                 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);                      \
    } else {                                                                \
        MPI_Recv(matrix[0], len, MPI_CHAR, infoProcess.id - 1, 0,           \
                 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);                      \
        MPI_Send(send_up, len, MPI_CHAR, infoProcess.id + 1, 0,             \
                 MPI_COMM_WORLD);                                           \
    }                                                                       \
                                                                            \
    /** down  */                                                            \
    if (infoProcess.id == ROOT) {                                           \
        MPI_Recv(matrix[image->height - 1], len, MPI_CHAR, ROOT + 1, 1,     \
                 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);                      \
    } else if (infoProcess.id == infoProcess.nrProcesses - 1) {             \
        MPI_Send(send_down, len, MPI_CHAR, infoProcess.id - 1, 1,           \
                 MPI_COMM_WORLD);                                           \
    } else {                                                                \
        MPI_Recv(matrix[image->height - 1], len, MPI_CHAR,                  \
                 infoProcess.id + 1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE); \
        MPI_Send(send_down, len, MPI_CHAR, infoProcess.id - 1, 1,           \
                 MPI_COMM_WORLD);                                           \
    }                                                                       \



#define ROOT 0


/******************************************************************************/

/**
 * funtia calculeaza numarul de lini care vor fi prelucrate de catre un proces
 * @param infoProcess
 */
static void calcul_limit(Info *infoProcess);

/**
 * procesul ROOT citeste din fisier toata imagine si dimensiunea, tipul
 * apoi trimite infomatia spre fiecare proces 
 * @param infoProcess
 * @param fileName
 */
static void readImage(Info *infoProcess, const char *fileName);

/**
 * toate procese != ROOT trimite portiuni din imagine catre ROOT care apoi
 * scrie imagimea rezultata
 * la final se elibereaza memoria
 * @param infoProcess
 * @param fileName
 */
static void writeImage(Info *infoProcess, const char *fileName);

/**
 * functia aplica filtrul img rezultatul fiind stocat in out
 * @param img
 * @param typeFilter
 * @param out
 */
static void apply_filter(Image *img, const char *typeFilter, Image *out);

/**
 * eliberarea matricei de pixeli
 * @param image
 */
static void free_matrix(Image *image);

/**
 * functia care trimite/primeste infomatia pentru un proces dupa aplicarea
 * unui filtru pe respectiva bucata atribuita acelui proces
 * @param infoProcess
 * @param image
 */
static void share_info(Info infoProcess, Image *image);


int main(int argc, char *argv[]) {
    Info infoProcess;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &infoProcess.id);
    MPI_Comm_size(MPI_COMM_WORLD, &infoProcess.nrProcesses);

    readImage(&infoProcess, argv[1]);

    Image in, out;
    in.matrix = infoProcess.img.matrix;
    in.width = infoProcess.img.width;
    in.typeImage = infoProcess.img.typeImage;
    in.height = (unsigned int) (infoProcess.end - infoProcess.start + 1);

    out.matrix = NULL;

    for (int i = 3; i < argc; i++) {
        apply_filter(&in, argv[i], &out);
        void *aux = in.matrix;
        in.matrix = out.matrix;
        out.matrix = aux;

        if (i != argc - 1)
            share_info(infoProcess, &in);
    }

    infoProcess.img.matrix = in.matrix;
    writeImage(&infoProcess, argv[2]);

    free_matrix(&out);

    MPI_Finalize();

    return 0;
}

static void calcul_limit(Info *infoProcess) {
    /** calcularea limitelor unde citeste */
    int med = infoProcess->img.height / infoProcess->nrProcesses;
    int rem = infoProcess->img.height % infoProcess->nrProcesses;
    infoProcess->start = infoProcess->id * med;
    infoProcess->end = infoProcess->start + med - 1;
    if (infoProcess->id < rem) {
        infoProcess->start += infoProcess->id;
        infoProcess->end = infoProcess->start + med;
    } else {
        infoProcess->start += rem;
        infoProcess->end += rem;
    }
    if (med == 0 && infoProcess->id < rem) {
        infoProcess->start = infoProcess->id;
        infoProcess->end = infoProcess->id;
    }

    if (infoProcess->start > infoProcess->end) {
        exit(EXIT_SUCCESS);
    }

    if (infoProcess->id != ROOT) {
        infoProcess->start--;
    }

    if (infoProcess->id != infoProcess->nrProcesses - 1) {
        infoProcess->end++;
    }

}

static void readImage(Info *infoProcess, const char *fileName) {
    if (infoProcess->id == ROOT) {
        FILE *input = fopen(fileName, "rb");
        assert(input != NULL);
        char type[3];
        int maxval;

        assert(fscanf(input, "%s", type) == 1);
        infoProcess->img.typeImage = type[1] - '0';
        assert(fscanf(input, "%u %u", &infoProcess->img.width,
                      &infoProcess->img.height) == 2);
        assert(fscanf(input, "%d", &maxval) == 1);
        fseek(input, 1, SEEK_CUR); /** Evitarea '\n' */

        int info_send[3];
        info_send[0] = infoProcess->img.height;
        info_send[1] = infoProcess->img.width;
        info_send[2] = infoProcess->img.typeImage;

        for (int i = 1; i < infoProcess->nrProcesses; i++) {
            MPI_Send(info_send, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        calcul_limit(infoProcess);

        size_t sizePixel;
        if (infoProcess->img.typeImage == grayscale) {
            sizePixel = sizeof(PixelGray);
        } else if (infoProcess->img.typeImage == color) {
            sizePixel = sizeof(PixelColor);
        } else exit(EXIT_FAILURE);

        unsigned char **buffer;
        buffer = malloc(infoProcess->img.height * sizeof(void *));
        assert(buffer != NULL);
        for (int i = 0; i < infoProcess->img.height; i++) {
            buffer[i] = malloc(infoProcess->img.width * sizePixel);
            assert(buffer[i] != NULL);
        }

        for (int i = 0; i < infoProcess->img.height; i++) {
            assert(fread(buffer[i], sizePixel, infoProcess->img.width, input) ==
                   infoProcess->img.width);
        }

        assert(fclose(input) == 0);

        for (int i = 1; i < infoProcess->nrProcesses; i++) {
            Info info_aux;
            info_aux.id = i;
            info_aux.nrProcesses = infoProcess->nrProcesses;
            info_aux.img.height = infoProcess->img.height;
            calcul_limit(&info_aux);

            for (int j = info_aux.start; j <= info_aux.end; j++) {
                MPI_Send(buffer[j], (int) (infoProcess->img.width * sizePixel),
                         MPI_CHAR, i, i, MPI_COMM_WORLD);
            }
        }

        unsigned char **matrix;
        int len = infoProcess->end - infoProcess->start + 1;
        matrix = malloc(len * sizeof(void *));
        assert(matrix != NULL);
        for (int i = 0; i < len; i++) {
            matrix[i] = malloc(infoProcess->img.width * sizePixel);
            assert(matrix[i] != NULL);
        }

        for (int i = 0; i < len; i++) {
            memcpy(matrix[i], buffer[i], infoProcess->img.width * sizePixel);
        }
        infoProcess->img.matrix = matrix;

        for (int i = 0; i < infoProcess->img.height; i++) {
            free(buffer[i]);
        }
        free(buffer);

    } else {
        int info_recv[3];
        MPI_Recv(info_recv, 3, MPI_INT, ROOT, 0, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);

        infoProcess->img.height = (unsigned int) info_recv[0];
        infoProcess->img.width = (unsigned int) info_recv[1];
        infoProcess->img.typeImage = info_recv[2];
        calcul_limit(infoProcess);

        size_t sizePixel;
        if (infoProcess->img.typeImage == grayscale) {
            sizePixel = sizeof(PixelGray);
        } else if (infoProcess->img.typeImage == color) {
            sizePixel = sizeof(PixelColor);
        } else exit(EXIT_FAILURE);

        unsigned char **matrix;
        int len = infoProcess->end - infoProcess->start + 1;
        matrix = malloc(len * sizeof(void *));
        assert(matrix != NULL);
        for (int i = 0; i < len; i++) {
            matrix[i] = malloc(infoProcess->img.width * sizePixel);
            assert(matrix[i] != NULL);
        }

        for (int i = 0; i < len; i++) {
            MPI_Recv(matrix[i], (int) (infoProcess->img.width * sizePixel),
                     MPI_CHAR, ROOT, infoProcess->id, MPI_COMM_WORLD,
                     MPI_STATUSES_IGNORE);
        }

        infoProcess->img.matrix = matrix;
    }
}

static void writeImage(Info *infoProcess, const char *fileName) {
    int lenDate;
    int start_send;
    int end_send;
    void *send;
    void *img_out;
    size_t sizePixel;
    int totalSize;

    if (infoProcess->img.typeImage == grayscale) {
        sizePixel = sizeof(PixelGray);
    } else if (infoProcess->img.typeImage == color) {
        sizePixel = sizeof(PixelColor);
    } else exit(EXIT_FAILURE);

    totalSize = (int) (infoProcess->img.width * infoProcess->img.height *
                       sizePixel);
    if (infoProcess->id == ROOT) {
        img_out = malloc((size_t) totalSize);
        assert(img_out != NULL);
    }

    if (infoProcess->id != ROOT) {
        start_send = infoProcess->start + 1;
    } else {
        start_send = infoProcess->start;
    }
    if (infoProcess->id != infoProcess->nrProcesses - 1) {
        end_send = infoProcess->end - 1;
    } else {
        end_send = infoProcess->end;
    }
    lenDate = (int) ((end_send - start_send + 1) * infoProcess->img.width *
                     sizePixel);

    /** infoanatia trimisa catre ROOT */
    send = malloc((size_t) lenDate);
    assert(send != NULL);

    /** copierea infoamatie in functie de tipul de imagine */
    if (infoProcess->img.typeImage == grayscale) {
        PixelGray **matrixGray = (PixelGray **) infoProcess->img.matrix;

        for (int i = start_send; i <= end_send; i++) {
            int j = i - start_send;
            int k = i - infoProcess->start;

            memcpy(send + j * infoProcess->img.width * sizePixel,
                   matrixGray[k], infoProcess->img.width * sizePixel);
        }

    } else {
        PixelColor **matrixColor = (PixelColor **) infoProcess->img.matrix;

        for (int i = start_send; i <= end_send; i++) {
            int j = i - start_send;
            int k = i - infoProcess->start;

            memcpy(send + j * infoProcess->img.width * sizePixel,
                   matrixColor[k], infoProcess->img.width * sizePixel);
        }
    }

    if (infoProcess->id == ROOT) {
        memcpy(img_out, send, lenDate);

        for (int i = 1; i < infoProcess->nrProcesses; i++) {
            ///
            /// recv[0] -> offset
            /// recv[1] -> len
            ///
            int recv[2]; // informatia primita
            MPI_Recv(recv, 2, MPI_INT, i, i, MPI_COMM_WORLD,
                     MPI_STATUSES_IGNORE);
            MPI_Recv(img_out + recv[0], recv[1], MPI_CHAR, i, i,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
    } else {
        int send_info[2];
        send_info[0] = (int) (start_send * sizePixel * infoProcess->img.width);
        send_info[1] = lenDate;

        MPI_Send(send_info, 2, MPI_INT, ROOT, infoProcess->id, MPI_COMM_WORLD);
        MPI_Send(send, send_info[1], MPI_CHAR, ROOT, infoProcess->id,
                 MPI_COMM_WORLD);
    }

    if (infoProcess->id == ROOT) {
        FILE *output = fopen(fileName, "wb+");

        // printare header
        assert(fprintf(output, "P%d\n", infoProcess->img.typeImage) != 0);
        assert(fprintf(output, "%u %u\n", infoProcess->img.width,
                       infoProcess->img.height) != 0);
        assert(fprintf(output, "%d\n", 255) != 0);

        assert(fwrite(img_out, sizeof(unsigned char), (size_t) totalSize,
                      output) == totalSize);

        assert(fclose(output) == 0);
        free(img_out);
    }

    free(send);
    int len = infoProcess->end - infoProcess->start + 1;
    if (infoProcess->img.typeImage == grayscale) {
        PixelGray **matrixGray = (PixelGray **) infoProcess->img.matrix;

        for (int i = 0; i < len; i++) {
            free(matrixGray[i]);
        }
        free(matrixGray);

    } else {
        PixelColor **matrixColor = (PixelColor **) infoProcess->img.matrix;

        for (int i = 0; i < len; i++) {
            free(matrixColor[i]);
        }
        free(matrixColor);
    }
}

static void apply_filter_color(int height, int width, PixelColor **matrix_in,
                               PixelColor **matrix_out, const float filtru[9]) {
    float r, g, b;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == 0 || j == 0 || i == height - 1 || j == width - 1) {
                matrix_out[i][j] = matrix_in[i][j];
            } else {
                r = CALC_PIXEL(matrix_in, i, j, filtru, R);
                g = CALC_PIXEL(matrix_in, i, j, filtru, G);
                b = CALC_PIXEL(matrix_in, i, j, filtru, B);
                matrix_out[i][j].r = (unsigned char) r;
                matrix_out[i][j].g = (unsigned char) g;
                matrix_out[i][j].b = (unsigned char) b;
            }
        }
    }
}

static void apply_filter_gray(int height, int width, PixelGray **matrix_in,
                              PixelGray **matrix_out, const float filtru[9]) {
    float pix;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i == 0 || j == 0 || i == height - 1 || j == width - 1) {
                matrix_out[i][j] = matrix_in[i][j];
            } else {
                pix = CALC_PIXEL(matrix_in, i, j, filtru, PIX);
                matrix_out[i][j].pix = (unsigned char) pix;
            }
        }
    }
}

static void apply_filter(Image *img, const char *typeFilter, Image *out) {
    float filtru[9];

    if (strcmp(typeFilter, "smooth") == 0) {
        float filtru_aux[9] = {1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9,
                               1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9};
        memcpy(&filtru, &filtru_aux, 9 * sizeof(float));
    } else if (strcmp(typeFilter, "blur") == 0) {
        float filtru_aux[9] = {1.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16,
                               4.0f / 16, 2.0f / 16, 1.0f / 16, 2.0f / 16,
                               1.0f / 16};
        memcpy(&filtru, &filtru_aux, 9 * sizeof(float));
    } else if (strcmp(typeFilter, "sharpen") == 0) {
        float filtru_aux[9] = {0, -2.0f / 3, 0, -2.0f / 3, 11.0f / 3,
                               -2.0f / 3, 0, -2.0f / 3, 0};
        memcpy(&filtru, &filtru_aux, 9 * sizeof(float));
    } else if (strcmp(typeFilter, "mean") == 0) {
        float filtru_aux[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
        memcpy(&filtru, &filtru_aux, 9 * sizeof(float));
    } else if (strcmp(typeFilter, "emboss") == 0) {
        float filtru_aux[9] = {0, 1, 0, 0, 0, 0, 0, -1, 0};
        memcpy(&filtru, &filtru_aux, 9 * sizeof(float));
    }

    out->typeImage = img->typeImage;
    out->height = img->height;
    out->width = img->width;

    if (img->typeImage == color) {
        PixelColor **matrix_out;

        if (out->matrix == NULL) {
            matrix_out = malloc(img->height * sizeof(PixelColor *));

            for (int i = 0; i < img->height; i++) {
                matrix_out[i] = malloc(img->width * sizeof(PixelColor));
            }

        } else {
            matrix_out = (PixelColor **) out->matrix;
        }

        apply_filter_color(img->height, img->width, (PixelColor **) img->matrix,
                           matrix_out, filtru);
        out->matrix = matrix_out;

    } else {
        PixelGray **matrix_out;

        if (out->matrix == NULL) {
            matrix_out = malloc(img->height * sizeof(PixelGray *));
            for (int i = 0; i < img->height; i++) {
                matrix_out[i] = malloc(img->width * sizeof(PixelGray));
            }

        } else {
            matrix_out = (PixelGray **) out->matrix;
        }

        apply_filter_gray(img->height, img->width, (PixelGray **) img->matrix,
                          matrix_out, filtru);
        out->matrix = matrix_out;
    }
}

static void free_matrix(Image *image) {

    if (image->typeImage == grayscale) {
        PixelGray **matrixGray = (PixelGray **) image->matrix;

        for (int i = 0; i < image->height; i++) {
            free(matrixGray[i]);
        }
        free(matrixGray);

    } else {
        PixelColor **matrixColor = (PixelColor **) image->matrix;

        for (int i = 0; i < image->height; i++) {
            free(matrixColor[i]);
        }
        free(matrixColor);
    }
}

static void share_info(Info infoProcess, Image *image) {

    if (infoProcess.nrProcesses == 1) {
        return;
    }

    size_t sizePixel;
    int len;
    void *send_up;
    void *send_down;

    if (infoProcess.img.typeImage == grayscale) {
        sizePixel = sizeof(PixelGray);
    } else if (infoProcess.img.typeImage == color) {
        sizePixel = sizeof(PixelColor);
    } else exit(EXIT_FAILURE);
    len = (int) (image->width * sizePixel);

    send_down = malloc((size_t) len);
    send_up = malloc((size_t) len);
    assert(send_down != NULL);
    assert(send_up != NULL);

    if (image->typeImage == grayscale) {
        SHARE(image, infoProcess, len, send_down, send_up, TYPEGRAY);
    } else {
        SHARE(image, infoProcess, len, send_down, send_up, TYPECOLOR);
    }

}