<template>
  <div class="d-flex flex-column ga-4">
    <v-card rounded="md">
      <v-card-text>
        <div class="config-title">群聊知识库</div>
        <div class="config-hint" style="line-height: 1.7;">
          每个群对每个知识库单独设定可见性。自动：对话时会把该库内容带进上下文；
          手动：不自动带入，但模型可以主动搜索；不可见：模型既不会自动拿到，也搜不到。
          未配置的群默认全部「手动」。保存后即时生效。
        </div>
      </v-card-text>
    </v-card>

    <div v-if="loading" class="d-flex justify-center py-10">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <v-empty-state
      v-else-if="!groups.length"
      icon="lucide:library"
      title="还没有群聊记录"
      text="群里产生过对话后会自动登记，也可以先在插件设置里手写群知识库映射"
    />

    <template v-else>
      <v-card v-for="group in groups" :key="group.group_id" rounded="md">
        <v-card-text class="config-section pb-2">
          <div class="d-flex align-center ga-3">
            <div class="flex-grow-1 min-w-0">
              <div class="config-title text-truncate">{{ group.group_name || '未命名群聊' }}</div>
              <div class="config-hint">群号 {{ group.group_id }}</div>
            </div>
            <v-chip size="small" color="info" variant="tonal">自动 {{ autoCount(group) }} 个</v-chip>
          </div>
        </v-card-text>

        <v-card-text class="pt-0 d-flex flex-column ga-2">
          <div v-for="domain in domains" :key="domain.key" class="d-flex align-center justify-space-between ga-3">
            <span class="text-body-2">{{ domain.label }}</span>
            <v-select
              :model-value="group.modes[domain.key]"
              :items="modeOptions"
              density="compact"
              hide-details
              variant="outlined"
              style="max-width: 140px;"
              :aria-label="domain.label + ' 的可见性'"
              @update:model-value="(value: string) => setMode(group, domain.key, value)"
            />
          </div>
        </v-card-text>
      </v-card>
    </template>

    <Transition name="fab">
      <v-btn
        v-if="dirty"
        class="save-fab"
        color="primary"
        variant="flat"
        size="large"
        rounded="pill"
        :loading="saving"
        prepend-icon="lucide:save"
        @click="save"
      >
        保存更改
      </v-btn>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'

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

const domains = ref<DomainMeta[]>([])
const groups = ref<GroupRow[]>([])
const loading = ref(true)
const saving = ref(false)
const baseline = ref('')

const modeOptions = [
  { title: '自动', value: 'auto' },
  { title: '手动', value: 'manual' },
  { title: '不可见', value: 'hidden' },
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
    toast.error(`加载群聊知识库失败: ${(e as Error).message}`)
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
    toast.success('已保存并即时生效')
  } catch (e) {
    toast.error(`保存失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.config-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}
.config-hint {
  font-size: 0.75rem;
  color: rgb(var(--v-theme-secondaryText));
  margin-top: 2px;
}

.save-fab {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 100;
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
