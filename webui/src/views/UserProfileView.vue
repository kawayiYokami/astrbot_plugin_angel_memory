<template>
  <div>
    <h2 class="text-h5 mb-4">👤 用户画像</h2>

    <v-row v-if="loading">
      <v-col cols="12" class="text-center">
        <v-progress-circular indeterminate color="primary" />
      </v-col>
    </v-row>

    <template v-else>
      <!-- 用户列表 -->
      <v-card v-if="!selectedUser">
        <v-card-title>已识别用户（{{ users.length }}）</v-card-title>
        <v-card-text>
          <v-alert v-if="!users.length" type="info" variant="tonal">
            暂无用户画像数据。用户画像会在对话过程中自动生成。
          </v-alert>
          <v-row v-else>
            <v-col v-for="user in users" :key="user.user_id" cols="12" sm="6" md="4">
              <v-card
                variant="outlined"
                class="pa-3 cursor-pointer"
                hover
                @click="selectUser(user)"
              >
                <div class="d-flex align-center ga-3 mb-2">
                  <v-avatar color="primary" size="40">
                    <span class="text-body-1">{{ (user.nickname || user.user_id).charAt(0) }}</span>
                  </v-avatar>
                  <div>
                    <div class="text-body-1 font-weight-medium">
                      {{ user.nickname || '未知昵称' }}
                    </div>
                    <div class="text-caption text-grey">ID: {{ user.user_id }}</div>
                  </div>
                </div>
                <div class="d-flex flex-wrap ga-1 mt-2">
                  <v-chip
                    v-for="(count, attr) in user.attributes"
                    :key="attr"
                    size="x-small"
                    :color="attrColor(attr as string)"
                    variant="tonal"
                  >
                    {{ attr }} ({{ count }})
                  </v-chip>
                </div>
                <div class="text-caption text-grey mt-2">共 {{ user.memory_count }} 条画像记忆</div>
              </v-card>
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>

      <!-- 用户详情 -->
      <template v-if="selectedUser">
        <v-btn
          variant="text"
          prepend-icon="mdi-arrow-left"
          class="mb-3"
          @click="selectedUser = null; profileMemories = []"
        >
          返回用户列表
        </v-btn>

        <v-card class="mb-4">
          <v-card-text class="d-flex align-center ga-4">
            <v-avatar color="primary" size="56">
              <span class="text-h6">{{ (selectedUser.nickname || selectedUser.user_id).charAt(0) }}</span>
            </v-avatar>
            <div>
              <div class="text-h6">{{ selectedUser.nickname || '未知昵称' }}</div>
              <div class="text-body-2 text-grey">用户 ID: {{ selectedUser.user_id }}</div>
              <div class="text-caption text-grey">共 {{ selectedUser.memory_count }} 条画像记忆</div>
            </div>
          </v-card-text>
        </v-card>

        <v-progress-linear v-if="profileLoading" indeterminate color="primary" class="mb-4" />

        <!-- 按属性分组展示 -->
        <template v-for="attr in attributeOrder" :key="attr">
          <v-card v-if="groupedMemories[attr]?.length" class="mb-4">
            <v-card-title>
              <v-chip :color="attrColor(attr)" size="small" class="mr-2">{{ attr }}</v-chip>
              {{ groupedMemories[attr].length }} 条
            </v-card-title>
            <v-card-text>
              <v-list density="compact">
                <v-list-item
                  v-for="mem in groupedMemories[attr]"
                  :key="mem.id"
                  class="mb-2"
                >
                  <v-card variant="tonal" class="pa-3">
                    <div class="d-flex align-center ga-2 mb-1">
                      <v-icon
                        :icon="mem.is_active ? 'mdi-star' : 'mdi-star-outline'"
                        :color="mem.is_active ? 'amber' : 'grey'"
                        size="small"
                      />
                      <v-chip size="x-small" :color="strengthColor(mem.strength)">
                        强度 {{ mem.strength }}
                      </v-chip>
                      <span class="text-caption text-grey">{{ formatTime(mem.updated_at) }}</span>
                    </div>
                    <div class="text-body-2 font-weight-medium">{{ mem.judgment }}</div>
                    <div v-if="mem.reasoning" class="text-caption text-grey mt-1">{{ mem.reasoning }}</div>
                    <div class="d-flex flex-wrap ga-1 mt-2">
                      <v-chip
                        v-for="tag in parseTags(mem.tags)"
                        :key="tag"
                        size="x-small"
                        :color="tagColor(tag)"
                        variant="tonal"
                      >
                        {{ tag }}
                      </v-chip>
                    </div>
                  </v-card>
                </v-list-item>
              </v-list>
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

const { apiGet } = useBridge()

const loading = ref(true)
const users = ref<any[]>([])
const selectedUser = ref<any>(null)
const profileMemories = ref<any[]>([])
const profileLoading = ref(false)

const attributeOrder = ['用户别名', '事实属性', '技能树', '关系图谱', '活跃项目']

const groupedMemories = computed(() => {
  const groups: Record<string, any[]> = {}
  for (const mem of profileMemories.value) {
    const attr = mem.attribute || '其他'
    if (!groups[attr]) groups[attr] = []
    groups[attr].push(mem)
  }
  return groups
})

function attrColor(attr: string): string {
  const map: Record<string, string> = {
    '用户别名': 'blue',
    '事实属性': 'green',
    '技能树': 'purple',
    '关系图谱': 'orange',
    '活跃项目': 'cyan',
  }
  return map[attr] || 'grey'
}

function strengthColor(s: number): string {
  if (s >= 80) return 'success'
  if (s >= 50) return 'primary'
  if (s >= 30) return 'warning'
  return 'error'
}

function formatTime(ts: number | null): string {
  if (!ts) return '-'
  let t = Number(ts)
  if (t > 1e11) t /= 1000
  return new Date(t * 1000).toLocaleString('zh-CN')
}

function parseTags(tags: string): string[] {
  if (!tags) return []
  return tags.split(',').map(t => t.trim()).filter(Boolean)
}

function tagColor(tag: string): string {
  const attrColors: Record<string, string> = {
    '用户别名': 'blue',
    '事实属性': 'green',
    '技能树': 'purple',
    '关系图谱': 'orange',
    '活跃项目': 'cyan',
  }
  if (attrColors[tag]) return attrColors[tag]
  if (/^\d{6,}$/.test(tag)) return 'grey'
  return 'primary'
}

async function selectUser(user: any) {
  selectedUser.value = user
  profileLoading.value = true
  try {
    const data: any = await apiGet('profiles/detail', { user_id: user.user_id })
    profileMemories.value = data.memories || []
  } catch (e) {
    console.error('加载用户画像失败:', e)
  } finally {
    profileLoading.value = false
  }
}

onMounted(async () => {
  try {
    const data: any = await apiGet('profiles')
    users.value = data.users || []
  } catch (e) {
    console.error('加载用户列表失败:', e)
  } finally {
    loading.value = false
  }
})
</script>
