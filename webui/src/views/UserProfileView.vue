<template>
  <div>
    <n-spin :show="loading">
      <template v-if="!loading">
        <!-- 用户列表 -->
        <n-card v-if="!selectedUser" :title="`已识别用户（${users.length}）`" embedded>
          <n-empty v-if="!users.length" description="暂无用户画像数据。用户画像会在对话过程中自动生成。" />
          <n-grid v-else :cols="3" :x-gap="12" :y-gap="12" responsive="screen" item-responsive>
            <n-grid-item v-for="user in users" :key="user.user_id" span="3 s:2 m:1">
              <n-card
                embedded
                hoverable
                class="user-card"
                @click="selectUser(user)"
              >
                <div class="user-head">
                  <n-avatar round :size="40" color="#7c4dff">
                    {{ (user.nickname || user.user_id).charAt(0) }}
                  </n-avatar>
                  <div class="user-meta">
                    <div class="user-name">{{ user.nickname || '未知昵称' }}</div>
                    <div class="user-id">ID: {{ user.user_id }}</div>
                  </div>
                </div>
                <div class="attr-chips">
                  <n-tag
                    v-for="(count, attr) in user.attributes"
                    :key="String(attr)"
                    size="small"
                    :type="attrType(String(attr))"
                    class="chip-item"
                    :bordered="false"
                  >
                    {{ attr }} ({{ count }})
                  </n-tag>
                </div>
                <div class="user-foot">共 {{ user.memory_count }} 条画像记忆</div>
              </n-card>
            </n-grid-item>
          </n-grid>
        </n-card>

        <!-- 用户详情 -->
        <template v-else>
          <n-button quaternary class="mb-3" @click="selectedUser = null; profileMemories = []">
            <template #icon><Icon icon="lucide:arrow-left" /></template>
            返回用户列表
          </n-button>

          <n-card embedded class="mb-4">
            <div class="detail-head">
              <n-avatar round :size="56" color="#7c4dff">
                {{ (selectedUser.nickname || selectedUser.user_id).charAt(0) }}
              </n-avatar>
              <div>
                <div class="detail-name">{{ selectedUser.nickname || '未知昵称' }}</div>
                <div class="detail-sub">用户 ID: {{ selectedUser.user_id }}</div>
                <div class="detail-sub">共 {{ selectedUser.memory_count }} 条画像记忆</div>
              </div>
            </div>
          </n-card>

          <n-spin :show="profileLoading">
            <!-- 按属性分组展示 -->
            <n-card
              v-for="attr in attributeOrder"
              :key="attr"
              v-show="groupedMemories[attr]?.length"
              embedded
              class="mb-4"
            >
              <template #header>
                <div class="group-header">
                  <n-tag size="small" :type="attrType(attr)" :bordered="false">{{ attr }}</n-tag>
                  <span class="group-count">{{ groupedMemories[attr]?.length || 0 }} 条</span>
                </div>
              </template>

              <div v-for="mem in groupedMemories[attr]" :key="mem.id" class="mem-card">
                <div class="mem-top">
                  <Icon
                    :icon="mem.is_active ? 'lucide:star' : 'lucide:star'"
                    :style="{ color: mem.is_active ? '#f0a020' : '#888', fontSize: '14px' }"
                  />
                  <n-tag size="small" :type="strengthType(mem.strength)" :bordered="false">
                    强度 {{ mem.strength }}
                  </n-tag>
                  <span class="mem-time">{{ formatTime(mem.updated_at) }}</span>
                </div>
                <div class="mem-judgment">{{ mem.judgment }}</div>
                <div v-if="mem.reasoning" class="mem-reasoning">{{ mem.reasoning }}</div>
                <div class="mem-tags">
                  <n-tag
                    v-for="tag in parseTags(mem.tags)"
                    :key="tag"
                    size="small"
                    :type="tagType(tag)"
                    class="chip-item"
                    :bordered="false"
                  >
                    {{ tag }}
                  </n-tag>
                </div>
              </div>
            </n-card>
          </n-spin>
        </template>
      </template>
    </n-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useBridge } from '@/composables/useBridge'
import { formatTime, parseTags } from '@/utils/format'

const { apiGet } = useBridge()

type TagType = 'default' | 'primary' | 'success' | 'info' | 'warning' | 'error'

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

function attrType(attr: string): TagType {
  const map: Record<string, TagType> = {
    用户别名: 'primary',
    事实属性: 'success',
    技能树: 'warning',
    关系图谱: 'error',
    活跃项目: 'info',
  }
  return map[attr] || 'default'
}

function strengthType(s: number): TagType {
  if (s >= 80) return 'success'
  if (s >= 50) return 'primary'
  if (s >= 30) return 'warning'
  return 'error'
}

function tagType(tag: string): TagType {
  if (Object.keys(attrType('')).length === 0) {
    // noop 防未用告警
  }
  const attrMap: Record<string, TagType> = {
    用户别名: 'primary',
    事实属性: 'success',
    技能树: 'warning',
    关系图谱: 'error',
    活跃项目: 'info',
  }
  if (attrMap[tag]) return attrMap[tag]
  if (/^\d{6,}$/.test(tag)) return 'default'
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

<style scoped>
.user-card {
  cursor: pointer;
}

.user-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}

.user-name {
  font-weight: 600;
  font-size: 14px;
}

.user-id {
  font-size: 12px;
  opacity: 0.6;
}

.attr-chips {
  display: flex;
  flex-wrap: wrap;
}

.chip-item {
  margin: 2px 4px 2px 0;
}

.user-foot {
  margin-top: 8px;
  font-size: 12px;
  opacity: 0.6;
}

.mb-3 {
  margin-bottom: 12px;
}

.mb-4 {
  margin-bottom: 16px;
}

.detail-head {
  display: flex;
  align-items: center;
  gap: 14px;
}

.detail-name {
  font-size: 18px;
  font-weight: 700;
}

.detail-sub {
  font-size: 13px;
  opacity: 0.65;
}

.group-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.group-count {
  font-size: 13px;
  opacity: 0.6;
}

.mem-card {
  background: var(--n-color-2, rgba(255, 255, 255, 0.04));
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 10px;
}

.mem-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.mem-time {
  font-size: 12px;
  opacity: 0.55;
}

.mem-judgment {
  font-weight: 600;
  font-size: 14px;
}

.mem-reasoning {
  font-size: 13px;
  opacity: 0.7;
  margin-top: 2px;
}

.mem-tags {
  display: flex;
  flex-wrap: wrap;
  margin-top: 6px;
}
</style>
