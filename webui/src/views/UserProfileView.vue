<template>
  <div class="d-flex flex-column ga-4">
    <div v-if="loading" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" />
    </div>

    <template v-else>
      <!-- 用户列表 -->
      <template v-if="!selectedUser">
        <v-empty-state
          v-if="!users.length"
          icon="lucide:users"
          title="还没有用户画像"
          text="对话过程中会自动积累画像记忆"
        />

        <v-row v-else>
          <v-col v-for="user in users" :key="`${user.platforms?.[0] || ''}-${user.user_id}`" cols="12" sm="6" md="4">
            <v-card rounded="md" hover class="user-card h-100" @click="selectUser(user)">
              <v-card-text>
                <div class="d-flex align-center ga-3 mb-2">
                  <v-avatar color="primary" size="40">{{ (user.nickname || user.user_id).charAt(0) }}</v-avatar>
                  <div class="min-w-0">
                    <div class="text-subtitle-2 font-weight-medium text-truncate">{{ user.nickname || '未知昵称' }}</div>
                    <div class="text-caption text-medium-emphasis text-truncate">ID: {{ user.user_id }}</div>
                  </div>
                </div>
                <div v-if="user.platforms?.length" class="d-flex flex-wrap ga-1 mb-2">
                  <v-chip v-for="plat in user.platforms" :key="String(plat)" size="x-small" variant="tonal">
                    {{ plat }}
                  </v-chip>
                </div>
                <div class="d-flex flex-wrap ga-1">
                  <v-chip
                    v-for="(count, attr) in user.attributes"
                    :key="String(attr)"
                    size="x-small"
                    :color="attrColor(String(attr))"
                    variant="tonal"
                  >
                    {{ attr }}（{{ count }}）
                  </v-chip>
                </div>
                <div class="text-caption text-medium-emphasis mt-2">共 {{ user.memory_count }} 条画像记忆</div>
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>

        <!-- 群聊账本 -->
        <template v-if="groups.length">
          <div class="config-title">群聊账本</div>
          <v-row>
            <v-col v-for="group in groups" :key="group.group_id" cols="12" sm="6" md="4">
              <v-card rounded="md">
                <v-card-text>
                  <div class="text-subtitle-2 font-weight-medium mb-1">{{ group.group_name || '未知群名' }}</div>
                  <div class="text-caption text-medium-emphasis">群号 {{ group.group_id }}</div>
                  <div class="text-caption text-medium-emphasis mt-1">最近活跃 {{ formatTime(group.last_seen_at) }}</div>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </template>
      </template>

      <!-- 用户详情 -->
      <template v-else>
        <div>
          <v-btn variant="text" prepend-icon="lucide:arrow-left" @click="backToList">返回用户列表</v-btn>
        </div>

        <v-card rounded="md">
          <v-card-text class="d-flex align-center ga-4">
            <v-avatar color="primary" size="56">{{ (selectedUser.nickname || selectedUser.user_id).charAt(0) }}</v-avatar>
            <div class="min-w-0">
              <div class="text-h6 font-weight-medium text-truncate">{{ selectedUser.nickname || '未知昵称' }}</div>
              <div class="text-caption text-medium-emphasis">用户 ID: {{ selectedUser.user_id }}</div>
              <div class="text-caption text-medium-emphasis">共 {{ selectedUser.memory_count }} 条画像记忆</div>
            </div>
          </v-card-text>
        </v-card>

        <div v-if="profileLoading" class="d-flex justify-center py-8">
          <v-progress-circular indeterminate color="primary" />
        </div>

        <v-empty-state
          v-else-if="!profileMemories.length"
          icon="lucide:user-x"
          title="这个用户还没有画像记忆"
          text="多聊几轮后会自动积累"
        />

        <template v-else>
          <v-card v-for="attr in attributeOrder" v-show="groupedMemories[attr]?.length" :key="attr" rounded="md">
            <v-card-text class="config-section pb-2">
              <div class="d-flex align-center ga-2">
                <v-chip size="small" :color="attrColor(attr)" variant="tonal">{{ attr }}</v-chip>
                <span class="text-caption text-medium-emphasis">{{ groupedMemories[attr]?.length || 0 }} 条</span>
              </div>
            </v-card-text>
            <v-card-text class="pt-0 d-flex flex-column ga-3">
              <div v-for="mem in groupedMemories[attr]" :key="mem.id" class="mem-card">
                <div class="d-flex align-center ga-2 mb-1 flex-wrap">
                  <v-icon
                    icon="lucide:star"
                    size="14"
                    :color="mem.is_active ? 'warning' : undefined"
                    :style="mem.is_active ? '' : 'opacity:.35'"
                  />
                  <v-chip size="x-small" :color="strengthColor(mem.strength)" variant="tonal">强度 {{ mem.strength }}</v-chip>
                  <span class="text-caption text-medium-emphasis">{{ formatTime(mem.updated_at) }}</span>
                </div>
                <div class="text-body-2 font-weight-medium">{{ mem.judgment }}</div>
                <div v-if="mem.reasoning" class="text-caption text-medium-emphasis mt-1">{{ mem.reasoning }}</div>
                <div v-if="parseTags(mem.tags).length" class="d-flex flex-wrap ga-1 mt-2">
                  <v-chip v-for="tag in parseTags(mem.tags)" :key="tag" size="x-small" variant="tonal">{{ tag }}</v-chip>
                </div>
              </div>
            </v-card-text>
          </v-card>
        </template>
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { toast } from '@/composables/useToast'
import { formatTime, parseTags } from '@/utils/format'

const { apiGet } = useBridge()

const loading = ref(true)
const users = ref<any[]>([])
const groups = ref<any[]>([])
const selectedUser = ref<any>(null)
const profileMemories = ref<any[]>([])
const profileLoading = ref(false)

const attributeOrder = ['用户别名', '事实属性', '技能树', '关系图谱', '活跃项目']

const groupedMemories = computed(() => {
  const out: Record<string, any[]> = {}
  for (const mem of profileMemories.value) {
    const attr = mem.attribute || '其他'
    if (!out[attr]) out[attr] = []
    out[attr].push(mem)
  }
  return out
})

function attrColor(attr: string): string {
  const map: Record<string, string> = {
    用户别名: 'primary',
    事实属性: 'success',
    技能树: 'warning',
    关系图谱: 'error',
    活跃项目: 'info',
  }
  return map[attr] || 'default'
}

function strengthColor(s: number): string {
  if (s >= 80) return 'success'
  if (s >= 50) return 'primary'
  if (s >= 30) return 'warning'
  return 'error'
}

async function selectUser(user: any) {
  selectedUser.value = user
  profileLoading.value = true
  try {
    const data: any = await apiGet('profiles/detail', { user_id: user.user_id })
    profileMemories.value = data.memories || []
  } catch (e) {
    toast.error(`加载用户画像失败: ${(e as Error).message}`)
  } finally {
    profileLoading.value = false
  }
}

function backToList() {
  selectedUser.value = null
  profileMemories.value = []
}

onMounted(async () => {
  try {
    const data: any = await apiGet('profiles')
    users.value = data.users || []
    groups.value = data.groups || []
  } catch (e) {
    toast.error(`加载用户列表失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.user-card {
  cursor: pointer;
}

.mem-card {
  background: rgba(var(--v-theme-on-surface), 0.04);
  border-radius: 8px;
  padding: 10px 12px;
}

.config-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: rgb(var(--v-theme-primaryText));
}
</style>
