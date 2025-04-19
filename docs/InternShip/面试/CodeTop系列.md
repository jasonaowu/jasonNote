---
article: false
title: CodeTop
order: 2
---

# 215. 数组中的第K个最大元素


## 快排
### 手写
O(nlogn)
- 哨兵划分： 以数组某个元素（一般选取首元素）为基准数，将所有小于基准数的元素移动至其左边，大于基准数的元素移动至其右边。
- 递归： 对 左子数组 和 右子数组 递归执行 哨兵划分，直至子数组长度为 1 时终止递归，即可完成对整个数组的排序。

```
import random
import sys
def quick_sort(nums, start, end):
    if start >= end:
        return 
    # 随机生成provix
    provix_idx = random.randint(start, end)
    left, right = start, end
    nums[left], nums[provix_idx] = nums[provix_idx], nums[left]
    x = provix = nums[left]
    while left < right:
        # 初始left 为空，所以先找不合法的right，更新left
        while left < right and nums[right] > x:
            right -= 1
        if left < right:
            nums[left] = nums[right]
            left += 1

        while left < right and nums[left] < x:
            left += 1
        if left < right:
            nums[right] = nums[left]
            right -= 1
    # 当left = right 的时候，left 为空
    nums[left] = provix
    quick_sort(nums, start, left - 1)  # left 上面放的是基准
    quick_sort(nums, left + 1, end)

lines = sys.stdin.read().splitlines()
n, k = map(int, lines[0].split())
# print(n, k)
nums = list(map(int, lines[1].split()))
quick_sort(nums, 0, n - 1)
print(nums[n - k])

```
### 调库
O(nlogn)
```
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 按从大到小排序后，输出idx = k - 1
        # nums.sort()
        # nums = nums[::-1]
        nums.sort(reverse = True)
        return nums[k-1]
```
### 快速选择算法--快排变种
O(n)</br>
不用递归处理左右两边了，而是每次缩小为一边</br>
快速选择只需要找到第 k 大或第 k 小的元素，而不是对整个数组进行排序。它的工作流程是通过分区操作递归查找包含目标元素的子数组。
```
 nums[left] = provix 
# quick_sort(nums, start, left - 1)
# quick_sort(nums, left + 1, end)
if left == len(nums) - k:
	return nums[left]
elif left < len(nums) - k:
	quick_sort(nums, left + 1, end)
else:
	quick_sort(nums, start, left - 1)
```
## 归并排序



## 堆排序