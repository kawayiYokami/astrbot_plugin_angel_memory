<template>
  <div>
    <h2 class="text-h5 mb-4">📌 系统总览</h2>

    <v-row v-if="loading">
      <v-col cols="12" class="text-center">
        <v-progress-circular indeterminate color="primary" />
      </v-col>
    </v-row>

    <template v-else>
      <v-row>
        <v-col cols="12" sm="6" md="3">
          <v-card variant="tonal" color="primary">
            <v-card-text class="text-center">
              <div class="text-h4">{{ overview.memory_count }}</div>
              <div class="text-caption">记忆总数</div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card variant="tonal" color="secondary">
            <v-card-text class="text-center">
              <div class="text-h4">{{ overview.global_tag_count }}</div>
              <div class="text-caption">全局标签</div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card variant="tonal" color="accent">
            <v-card-text class="text-center">
              <div class="text-h4">{{ overview.note_index_count }}</div>
              <div class="text-caption">笔记索引</div>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" sm="6" md="3">
          <v-card variant="tonal" :color="overview.has_providers ? 'success' : 'warning'">
            <v-card-text class="text-center">
              <v-icon :icon="overview.has_providers ? 'mdi-check-circle' : 'mdi-alert'" size="32" />
              <div class="text-caption mt-1">{{ overview.has_providers ? '提供商就绪' : '无提供商' }}</div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <v-row class="mt-4">
        <v-col cols="12" md="6">
          <v-card>
            <v-card-title>配置信息</v-card-title>
            <v-card-text>
              <v-list density="compact">
                <v-list-item>
                  <template v-slot:prepend><v-icon icon="mdi-chip" size="small" /></template>
                  <v-list-item-title>嵌入提供商</v-list-item-title>
                  <v-list-item-subtitle>{{ overview.provider_id }}</v-list-item-subtitle>
                </v-list-item>
                <v-list-item>
                  <template v-slot:prepend><v-icon icon="mdi-robot" size="small" /></template>
                  <v-list-item-title>LLM 提供商</v-list-item-title>
                  <v-list-item-subtitle>{{ overview.llm_provider_id }}</v-list-item-subtitle>
                </v-list-item>
                <v-list-item>
                  <template v-slot:prepend><v-icon icon="mdi-folder" size="small" /></template>
                  <v-list-item-title>索引目录</v-list-item-title>
                  <v-list-item-subtitle class="text-truncate">{{ overview.index_dir }}</v-list-item-subtitle>
                </v-list-item>
                <v-list-item>
                  <template v-slot:prepend><v-icon icon="mdi-database" size="small" /></template>
                  <v-list-item-title>向量索引</v-list-item-title>
                  <v-list-item-subtitle>{{ overview.has_vector_db ? '可用' : '不可用' }}</v-list-item-subtitle>
                </v-list-item>
              </v-list>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" md="6">
          <v-card>
            <v-card-title>Scope 列表</v-card-title>
            <v-card-text>
              <v-chip
                v-for="scope in overview.scopes"
                :key="scope"
                class="ma-1"
                color="primary"
                variant="outlined"
                size="small"
              >
                {{ scope }}
              </v-chip>
              <div v-if="!overview.scopes?.length" class="text-grey">暂无 scope</div>
            </v-card-text>
          </v-card>

          <v-card class="mt-4" v-if="overview.vector_collections?.length">
            <v-card-title>向量集合</v-card-title>
            <v-card-text>
              <v-chip
                v-for="col in overview.vector_collections"
                :key="col"
                class="ma-1"
                color="secondary"
                variant="outlined"
                size="small"
              >
                {{ col }}
              </v-chip>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'

const { apiGet } = useBridge()

const loading = ref(true)
const overview = ref<Record<string, any>>({})

onMounted(async () => {
  try {
    overview.value = await apiGet('overview')
  } catch (e) {
    console.error('加载总览失败:', e)
  } finally {
    loading.value = false
  }
})
</script>
