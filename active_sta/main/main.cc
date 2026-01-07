#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_idf_version.h"
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
#include "esp_flash.h"
#include "spi_flash_mmap.h"
#else
#include "esp_spi_flash.h"
#endif
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "lwip/err.h"
#include "lwip/sys.h"

#include "../../_components/nvs_component.h"
#include "../../_components/sd_component.h"
#include "../../_components/csi_component.h"
#include "../../_components/time_component.h"
#include "../../_components/input_component.h"
#include "../../_components/sockets_component.h"

/*
 * The examples use WiFi configuration that you can set via 'idf.py menuconfig'.
 *
 * If you'd rather not, just change the below entries to strings with
 * the config you want - ie #define ESP_WIFI_SSID "mywifissid"
 */
#define ESP_WIFI_SSID      CONFIG_ESP_WIFI_SSID
#define ESP_WIFI_PASS      CONFIG_ESP_WIFI_PASSWORD

#ifdef CONFIG_WIFI_CHANNEL
#define WIFI_CHANNEL CONFIG_WIFI_CHANNEL
#else
#define WIFI_CHANNEL 6
#endif

#ifdef CONFIG_SHOULD_COLLECT_CSI
#define SHOULD_COLLECT_CSI 1
#else
#define SHOULD_COLLECT_CSI 0
#endif

#ifdef CONFIG_SHOULD_COLLECT_ONLY_LLTF
#define SHOULD_COLLECT_ONLY_LLTF 1
#else
#define SHOULD_COLLECT_ONLY_LLTF 0
#endif

#ifdef CONFIG_SEND_CSI_TO_SERIAL
#define SEND_CSI_TO_SERIAL 1
#else
#define SEND_CSI_TO_SERIAL 0
#endif

#ifdef CONFIG_SEND_CSI_TO_SD
#define SEND_CSI_TO_SD 1
#else
#define SEND_CSI_TO_SD 0
#endif

/* FreeRTOS event group to signal when we are connected*/
static EventGroupHandle_t s_wifi_event_group;

/* The event group allows multiple bits for each event, but we only care about one event
 * - are we connected to the AP with an IP? */
const int WIFI_CONNECTED_BIT = BIT0;

static const char *TAG = "Active CSI collection (Station)";

// Forward decl (used by TELLO connect task)
bool is_wifi_connected();

// ---- TELLO Fast Active Scan (SSID prefix) ----
static const char *TELLO_SSID_PREFIX = "TELLO-";
static const size_t TELLO_SSID_PREFIX_LEN = sizeof("TELLO-") - 1;
static const uint16_t TELLO_SCAN_ACTIVE_MIN_MS = 30;
static const uint16_t TELLO_SCAN_ACTIVE_MAX_MS = 60;

static TaskHandle_t s_tello_connect_task_handle = NULL;

static esp_err_t scan_best_tello_ap(wifi_ap_record_t *out_best_ap, bool *out_found) {
    if (!out_best_ap || !out_found) {
        return ESP_ERR_INVALID_ARG;
    }
    *out_found = false;

    wifi_scan_config_t scan_config = {};
    scan_config.scan_type = WIFI_SCAN_TYPE_ACTIVE;
    scan_config.show_hidden = false;
    scan_config.scan_time.active.min = TELLO_SCAN_ACTIVE_MIN_MS;
    scan_config.scan_time.active.max = TELLO_SCAN_ACTIVE_MAX_MS;

    esp_err_t err = esp_wifi_scan_start(&scan_config, true);
    if (err != ESP_OK) {
        return err;
    }

    uint16_t ap_count = 0;
    err = esp_wifi_scan_get_ap_num(&ap_count);
    if (err != ESP_OK || ap_count == 0) {
        return err;
    }

    wifi_ap_record_t *ap_list = (wifi_ap_record_t *) malloc(sizeof(wifi_ap_record_t) * ap_count);
    if (!ap_list) {
        return ESP_ERR_NO_MEM;
    }

    uint16_t number = ap_count;
    err = esp_wifi_scan_get_ap_records(&number, ap_list);
    if (err != ESP_OK) {
        free(ap_list);
        return err;
    }

    int best_idx = -1;
    int best_rssi = -127;
    for (int i = 0; i < (int) number; i++) {
        if (strncmp((const char *) ap_list[i].ssid, TELLO_SSID_PREFIX, TELLO_SSID_PREFIX_LEN) != 0) {
            continue;
        }
        if (best_idx < 0 || ap_list[i].rssi > best_rssi) {
            best_idx = i;
            best_rssi = ap_list[i].rssi;
        }
    }

    if (best_idx >= 0) {
        memcpy(out_best_ap, &ap_list[best_idx], sizeof(wifi_ap_record_t));
        *out_found = true;
    }

    free(ap_list);
    return ESP_OK;
}

static void tello_connect_task(void *pvParameters) {
    (void) pvParameters;

    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // If already connected, ignore this trigger.
        if (is_wifi_connected()) {
            continue;
        }

        wifi_ap_record_t best_ap = {};
        bool found = false;
        esp_err_t err = scan_best_tello_ap(&best_ap, &found);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "TELLO scan failed: %s (retrying)", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(200));
            xTaskNotifyGive(s_tello_connect_task_handle);
            continue;
        }
        if (!found) {
            ESP_LOGW(TAG, "No TELLO-* AP found (retrying)");
            vTaskDelay(pdMS_TO_TICKS(200));
            xTaskNotifyGive(s_tello_connect_task_handle);
            continue;
        }

        wifi_config_t wifi_config = {};
        strlcpy((char *) wifi_config.sta.ssid, (const char *) best_ap.ssid, sizeof(wifi_config.sta.ssid));
        // Tello is Open AP (no password)
        wifi_config.sta.password[0] = '\0';
        wifi_config.sta.channel = best_ap.primary; // lock channel for fast connect
        wifi_config.sta.bssid_set = 1;
        memcpy(wifi_config.sta.bssid, best_ap.bssid, sizeof(best_ap.bssid));
        wifi_config.sta.threshold.authmode = WIFI_AUTH_OPEN;

        ESP_LOGI(TAG, "Best TELLO AP: SSID:%s CH:%u RSSI:%d BSSID:" MACSTR,
                 (const char *) best_ap.ssid, best_ap.primary, best_ap.rssi, MAC2STR(best_ap.bssid));

        ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
        err = esp_wifi_connect();
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "esp_wifi_connect failed: %s (retrying)", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(200));
            xTaskNotifyGive(s_tello_connect_task_handle);
        }
    }
}

esp_err_t _http_event_handle(esp_http_client_event_t *evt) {
    switch (evt->event_id) {
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
            if (!esp_http_client_is_chunked_response(evt->client)) {
                if (!real_time_set) {
                    char *data = (char *) malloc(evt->data_len + 1);
                    strncpy(data, (char *) evt->data, evt->data_len);
                    data[evt->data_len + 1] = '\0';
                    time_set(data);
                    free(data);
                }
            }
            break;
        default:
            break;
    }
    return ESP_OK;
}

//// en_sys_seq: see https://github.com/espressif/esp-idf/blob/master/docs/api-guides/wifi.rst#wi-fi-80211-packet-send for details
esp_err_t esp_wifi_80211_tx(wifi_interface_t ifx, const void *buffer, int len, bool en_sys_seq);

static void event_handler(void* arg, esp_event_base_t event_base,
                          int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        if (s_tello_connect_task_handle) {
            xTaskNotifyGive(s_tello_connect_task_handle);
        }
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected. Re-scanning TELLO and retrying...");
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        if (s_tello_connect_task_handle) {
            xTaskNotifyGive(s_tello_connect_task_handle);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

bool is_wifi_connected() {
    return (xEventGroupGetBits(s_wifi_event_group) & WIFI_CONNECTED_BIT);
}

void station_init() {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());

    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    // Create the TELLO connect manager before starting WiFi, so it can catch WIFI_EVENT_STA_START immediately.
    xTaskCreatePinnedToCore(&tello_connect_task, "tello_connect_task",
                            4096, NULL, 10, &s_tello_connect_task_handle, 0);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    esp_wifi_set_ps(WIFI_PS_NONE);

    ESP_LOGI(TAG, "Fast Active Scan: prefix=%s active=%ums~%ums/ch",
             TELLO_SSID_PREFIX, TELLO_SCAN_ACTIVE_MIN_MS, TELLO_SCAN_ACTIVE_MAX_MS);
}

TaskHandle_t xHandle = NULL;

void vTask_socket_transmitter_sta_loop(void *pvParameters) {
    for (;;) {
        socket_transmitter_sta_loop(&is_wifi_connected);
    }
}

void config_print() {
    printf("\n\n\n\n\n\n\n\n");
    printf("-----------------------\n");
    printf("ESP32 CSI Tool Settings\n");
    printf("-----------------------\n");
    printf("PROJECT_NAME: %s\n", "ACTIVE_STA");
    printf("CONFIG_ESPTOOLPY_MONITOR_BAUD: %d\n", CONFIG_ESPTOOLPY_MONITOR_BAUD);
    printf("CONFIG_ESP_CONSOLE_UART_BAUDRATE: %d\n", CONFIG_ESP_CONSOLE_UART_BAUDRATE);
    printf("IDF_VER: %s\n", IDF_VER);
    printf("-----------------------\n");
    printf("WIFI_CHANNEL: %d\n", WIFI_CHANNEL);
    printf("ESP_WIFI_SSID: %s\n", ESP_WIFI_SSID);
    printf("ESP_WIFI_PASSWORD: %s\n", ESP_WIFI_PASS);
    printf("PACKET_RATE: %i\n", CONFIG_PACKET_RATE);
    printf("SHOULD_COLLECT_CSI: %d\n", SHOULD_COLLECT_CSI);
    printf("SHOULD_COLLECT_ONLY_LLTF: %d\n", SHOULD_COLLECT_ONLY_LLTF);
    printf("SEND_CSI_TO_SERIAL: %d\n", SEND_CSI_TO_SERIAL);
    printf("SEND_CSI_TO_SD: %d\n", SEND_CSI_TO_SD);
    printf("-----------------------\n");
    printf("\n\n\n\n\n\n\n\n");
}

extern "C" void app_main() {
    config_print();
    nvs_init();
    sd_init();
    station_init();
    csi_init((char *) "STA");

#if !(SHOULD_COLLECT_CSI)
    printf("CSI will not be collected. Check `idf.py menuconfig  # > ESP32 CSI Tool Config` to enable CSI");
#endif

    xTaskCreatePinnedToCore(&vTask_socket_transmitter_sta_loop, "socket_transmitter_sta_loop",
                            10000, (void *) &is_wifi_connected, 100, &xHandle, 1);
}
