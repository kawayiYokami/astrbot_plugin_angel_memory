<template>
  <div class="group-knowledge-page">
    <n-spin :show="loading">
      <n-space vertical :size="16">
        <n-card embedded>
          <div class="head">
            <p class="title">群聊知识库</p>
            <p class="desc">
              每个群对每个知识库单独设定可见性。自动：对话时会把该库内容注入上下文；
              手动：不自动注入，但模型可以主动搜索；不可见：模型既不会自动拿到，也搜不到。
              未配置的群默认全部「手动」。保存后即时生效。
            </p>
          </div>
        </n-card>

        <n-empty
          v-if="!groups.length"
          description="群账本里还没有群聊记录。群里产生过对话后会自动登记，也可以先在插件设置里手写 group_knowledge_map。"
        />

        <n-card v-for="group in groups" :key="group.group_id" embedded>
          <div class="group-head">
            <div class="group-meta">
              <div class="group-name">{{ group.group_name || '未命名群聊' }}</div>
              <div class="group-id">群号 {{ group.group_id }}</div>
            </div>
            <n-tag :bordered="false" size="small" type="info">
              自动 {{ autoCount(group) }} 个
            </n-tag>
          </div>

          <div class="domain-list">
            <div v-for="domain in domains" :key="domain.key" class="domain-row">
              <div class="domain-name">
                <span class="domain-label">{{ domain.label }}</span>
                <span class="domain-key">{{ domain.key }}</span>
              </div>
              <n-select
                :value="group.modes[domain.key]"
                :options="modeOptions"
                size="small"
                class="mode-select"
                @update:value="(value: string) => setMode(group, domain.key, value)"
              />
            </div>
          </div>
        </n-card>

        <Transition name="fab">
          <n-button
            v-if="dirty"
            class="save-fab"
            type="primary"
            round
            size="large"
            :loading="saving"
            @click="save"
          >
            <template #icon><Icon icon="lucide:save" /></template>
            保存更改
          </n-button>
        </Transition>
      </n-space>
    </n-spin>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  NButton,
  NCard,
  NEmpty,
  NSelect,
  NSpace,
  NSpin,
  NTag,
  useMessage,
} from 'naive-ui'
import { Icon } from '@iconify/vue'
import { useBridge } from '@/composables/useBridge'

interface DomainMeta {
  key: string
  label: string
}

interface GroupRow {
  group_id: string
  group_name: string
  modes: Record<string, string>
}

const { apiGet, apiPost } = useBridge()
const message = useMessage()

const domains = ref<DomainMeta[]>([])
const groups = ref<GroupRow[]>([])
const loading = ref(true)
const saving = ref(false)
const baseline = ref('')

const modeOptions = [
  { label: '自动', value: 'auto' },
  { label: '手动', value: 'manual' },
  { label: '不可见', value: 'hidden' },
]

const dirty = computed(
  () => !loading.value && JSON.stringify(groups.value) !== baseline.value,
)

function autoCount(group: GroupRow): number {
  return Object.values(group.modes || {}).filter((mode) => mode === 'auto').length
}

function setMode(group: GroupRow, domainKey: string, value: string) {
  group.modes = { ...group.modes, [domainKey]: value }
}

async function load() {
  loading.value = true
  try {
    const data: any = await apiGet('group-knowledge')
    domains.value = data?.domains || []
    groups.value = (data?.groups || []).map((row: GroupRow) => ({
      group_id: row.group_id,
      group_name: row.group_name || '',
      modes: { ...(row.modes || {}) },
    }))
    baseline.value = JSON.stringify(groups.value)
  } catch (e) {
    message.error(`加载群聊知识库失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
}

async function save() {
  saving.value = true
  try {
    await apiPost('group-knowledge/save', {
      groups: groups.value.map((row) => ({
        group_id: row.group_id,
        modes: row.modes,
      })),
    })
    baseline.value = JSON.stringify(groups.value)
    message.success('已保存并即时生效')
  } catch (e) {
    message.error(`保存失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.head .title {
  margin: 0 0 6px;
  font-size: 16px;
  font-weight: 600;
}

.head .desc {
  margin: 0;
  font-size: 13px;
  opacity: 0.75;
  line-height: 1.6;
}

.group-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.group-name {
  font-size: 15px;
  font-weight: 600;
}

.group-id {
  font-size: 12px;
  opacity: 0.6;
  margin-top: 2px;
}

.domain-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.domain-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.domain-name {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.domain-label {
  font-size: 13px;
}

.domain-key {
  font-size: 11px;
  opacity: 0.45;
}

.mode-select {
  width: 140px;
}

.save-fab {
  position: fixed;
  right: 32px;
  bottom: 32px;
  z-index: 10;
}

.fab-enter-active,
.fab-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fab-enter-from,
.fab-leave-to {
  opacity: 0;
  transform: translateY(8px);
}
</style>
