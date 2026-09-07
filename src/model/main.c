/*
 * Mini-GPT CLI: обучение и генерация кода.
 * Использование:
 *   minigpt train <data_dir> [epochs] [seq_len]
 *   minigpt generate <model.bin> [temp] [max_tokens]
 *   minigpt interactive <model.bin>
 */
#include "mini_gpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

static void print_divider(void) {
    for (int i = 0; i < 40; i++) printf("=");
    printf("\n");
}

static void usage(void) {
    printf("Mini-GPT - языковая модель для генерации кода на чистом C\n\n");
    printf("Использование:\n");
    printf("  minigpt train <data> [epochs] [seq_len] [model.bin]\n");
    printf("      data: путь к папке с кодом ИЛИ текстовому файлу\n");
    printf("      пример: minigpt train data/training/code_files 50 128 model.bin\n\n");
    printf("  minigpt generate <model.bin> [temp] [max_tokens] \"<prompt>\"\n");
    printf("      temp: температура (0.0 - детерминировано, 1.0 - случайно)\n");
    printf("      пример: minigpt generate model.bin 0.8 500 \"def factorial\"\n\n");
    printf("  minigpt interactive <model.bin>\n");
    printf("      Режим интерактивной генерации кода\n\n");
    printf("  minigpt eval <model.bin> <file>\n");
    printf("      Оценка перплексии на файле\n");
}

static int is_directory(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (st.st_mode & S_IFDIR) != 0;
}

/* Γенерация из промпта (строка) */
static void generate_from_prompt(Model *m, const char *prompt,
                                 float temp, int max_tokens) {
    int plen = (int)strlen(prompt);
    if (plen > MAX_SEQ_LEN) plen = MAX_SEQ_LEN;

    int *tokens = (int*)malloc(plen * sizeof(int));
    for (int i = 0; i < plen; i++)
        tokens[i] = (unsigned char)prompt[i];

    int *out = (int*)malloc(max_tokens * sizeof(int));
    int n = model_generate(m, tokens, plen, max_tokens, temp, out);

    printf("%s", prompt);
    for (int i = 0; i < n; i++)
        putchar(out[i]);
    printf("\n");

    free(tokens); free(out);
}

int cmd_train(int argc, char **argv) {
    if (argc < 3) {
        printf("  [!] Укажите путь к данным\n");
        return 1;
    }
    const char *data_path = argv[2];
    int epochs = argc > 3 ? atoi(argv[3]) : 20;
    int seq_len = argc > 4 ? atoi(argv[4]) : BLOCK_SIZE;
    const char *model_path = argc > 5 ? argv[5] : "model.bin";

    printf("Mini-GPT: обучение\n");
    printf("  Данные:  %s\n", data_path);
    printf("  Эпох:    %d\n", epochs);
    printf("  Seq len: %d\n", seq_len);
    printf("  Модель:  %s\n", model_path);
    Model m;
    model_init(&m);
    printf("  Параметры: %d\n\n", model_nparams(&m));

    int data_len = 0;
    int *data;

    if (is_directory(data_path)) {
        printf("  Загрузка файлов из папки...\n");
        data = load_directory(data_path, &data_len);
    } else {
        printf("  Загрузка файла...\n");
        data = load_text_file(data_path, &data_len);
    }

    if (!data || data_len < 100) {
        printf("  [!] Не удалось загрузить данные из %s (len=%d)\n", data_path, data_len);
        model_free(&m);
        return 1;
    }

    printf("  Загружено: %d байт\n\n", data_len);

    TrainConfig cfg = train_config_default();
    cfg.epochs = epochs;
    cfg.max_seq_len = seq_len;
    cfg.save_path = model_path;
    cfg.print_every = 50;
    cfg.save_every = 1000;

    train_on_data(&m, data, data_len, &cfg);

    printf("\n  Тест генерации после обучения:\n");
    print_divider();
    generate_from_prompt(&m, "def factorial(n):", 0.8f, 300);

    free(data);
    model_free(&m);
    return 0;
}

int cmd_generate(int argc, char **argv) {
    if (argc < 3) {
        printf("  [!] Укажите модель\n");
        return 1;
    }
    const char *model_path = argv[2];
    float temp = argc > 3 ? (float)atof(argv[3]) : 0.8f;
    int max_tokens = argc > 4 ? atoi(argv[4]) : 500;
    const char *prompt = argc > 5 ? argv[5] : "def ";

    Model m;
    model_init(&m);
    if (model_load(&m, model_path) != 0) {
        printf("  [!] Не удалось загрузить модель %s\n", model_path);
        model_free(&m);
        return 1;
    }

    printf("  Модель: %s (шаг %d)\n", model_path, m.step);
    printf("  Temp: %.2f | Max tokens: %d\n", temp, max_tokens);
    print_divider();

    generate_from_prompt(&m, prompt, temp, max_tokens);

    model_free(&m);
    return 0;
}

int cmd_interactive(int argc, char **argv) {
    if (argc < 3) {
        printf("  [!] Укажите модель\n");
        return 1;
    }
    const char *model_path = argv[2];

    Model m;
    model_init(&m);
    if (model_load(&m, model_path) != 0) {
        printf("  [!] Не удалось загрузить модель %s\n", model_path);
        model_free(&m);
        return 1;
    }

    printf("Interactive mode. Введите 'quit' для выхода.\n");
    printf("Введите 'temp 0.8' для изменения температуры.\n\n");

    float temp = 0.8f;
    char line[2048];
    while (1) {
        printf(">>> ");
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\n")] = 0;

        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
        if (strncmp(line, "temp ", 5) == 0) {
            temp = (float)atof(line + 5);
            printf("  Температура: %.2f\n", temp);
            continue;
        }
        if (strlen(line) == 0) continue;

        generate_from_prompt(&m, line, temp, 800);
        printf("\n");
    }

    model_free(&m);
    return 0;
}

int cmd_eval(int argc, char **argv) {
    if (argc < 4) {
        printf("  [!] minigpt eval <model.bin> <file>\n");
        return 1;
    }
    const char *model_path = argv[2];
    const char *file_path = argv[3];

    Model m;
    model_init(&m);
    if (model_load(&m, model_path) != 0) {
        printf("  [!] Не удалось загрузить модель %s\n", model_path);
        model_free(&m);
        return 1;
    }

    int len = 0;
    int *data = load_text_file(file_path, &len);
    if (!data || len < 2) {
        printf("  [!] Не удалось загрузить %s\n", file_path);
        model_free(&m);
        return 1;
    }

    int T = BLOCK_SIZE;
    float total_loss = 0;
    int count = 0;

    for (int start = 0; start + T < len; start += T / 2) {
        float loss = model_forward(&m, data + start, T, T - 1);
        total_loss += loss;
        count++;
    }

    float avg_loss = count ? total_loss / count : 0;
    printf("  Perplexity: %.4f (avg loss: %.4f over %d windows)\n",
           expf(avg_loss), avg_loss, count);

    free(data);
    model_free(&m);
    return 0;
}

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));

    if (argc < 2) {
        usage();
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        return cmd_train(argc, argv);
    } else if (strcmp(argv[1], "generate") == 0) {
        return cmd_generate(argc, argv);
    } else if (strcmp(argv[1], "interactive") == 0) {
        return cmd_interactive(argc, argv);
    } else if (strcmp(argv[1], "eval") == 0) {
        return cmd_eval(argc, argv);
    } else {
        usage();
        return 1;
    }
}
