#ifndef _MIKUMARI_CACHE_H_
#define _MIKUMARI_CACHE_H_
#include <mutex>
#include <vector>

namespace mikumari {

template<typename T> class LinkedList;

template<typename T> class LinkedListElement {
public:
	LinkedListElement<T>* next = nullptr;
	LinkedListElement<T>* prev = nullptr;
	LinkedList<T>* container = nullptr;
	T data;
	LinkedListElement(T &data) : data(data) {}
};

template<typename T> class LinkedList {
public:
	LinkedListElement<T>* head = nullptr;
	LinkedListElement<T>* tail = nullptr;

	~LinkedList() {
		while (popHead() != nullptr);
	}

	bool isEmpty() {
		return head==nullptr;
	}

	int size() {
		LinkedListElement<T>* next = head;
		int count = 0;
		while (next != nullptr) {
			next = next->next;
			count++;
		}
		return count;
	}

	T popHead() {
		if (isEmpty()) return nullptr;
		LinkedListElement<T>* elem = head;
		if (elem != nullptr) {
			head = head->next;
			if (head != nullptr) head->prev = nullptr;
		}
		if (head == nullptr && tail == elem) tail = nullptr;
		T data = elem->data;
		delete elem;
		return data;
	}

	T popTail() {
		if (isEmpty()) return nullptr;
		LinkedListElement<T>* elem = tail;
		if (elem != nullptr) {
			tail = tail->prev;
			if (tail != nullptr) tail->next = nullptr;
		}
		if (tail == nullptr && head == elem) head = nullptr;
		T data = elem->data;
		delete elem;
		return data;
	}

	bool remove(LinkedListElement<T>* element) {
		if (element == nullptr || element->container != this) return false;
		if (element->next != nullptr) element->next->prev = element->prev;
		else if (tail == element) tail = element->prev;
		if (element->prev != nullptr) element->prev->next = element->next;
		else if (head == element) head = element->next;
		delete element;
		return true;
	}

	LinkedListElement<T>* pushBack(T data) {
		LinkedListElement<T>* element = new LinkedListElement<T>(data);
		if (head == nullptr) {
			head = element;
			tail = element;
			element->next = nullptr;
			element->prev = nullptr;
			element->container = this;
		} else {
			element->prev = tail;
			tail->next = element;
			tail = element;
			element->container = this;
		}
		return element;
	}
};

struct Page;

struct Allocation {
  bool evicted = false;
  int usage_count = 0;
  std::vector<Page*> pages;
  std::vector<char*> page_pointers;
  std::function<void(void)> eviction_callback;
  LinkedListElement<std::shared_ptr<Allocation>>* list_position = nullptr;
};

struct Page {
	char* ptr;
	std::shared_ptr<Allocation> current_allocation;
};

class PageCache {
private:
	std::recursive_mutex mutex;
	const bool allowEvictions;
	std::vector<char*> baseptrs;

public:
	const size_t size, page_size;
	const unsigned n_pages;
	LinkedList<Page*> freePages;
	LinkedList<std::shared_ptr<Allocation>> lockedAllocations, unlockedAllocations;

	PageCache(char* baseptr, size_t total_size, size_t page_size, const bool allowEvictions = true);
	PageCache(std::vector<std::pair<char*, size_t>> baseptrs, size_t total_size, size_t page_size, const bool allowEvictions = true);

	virtual ~PageCache() = default;

	/*
	Locks the allocation if it hasn't been evicted
	*/
	bool trylock(std::shared_ptr<Allocation> allocation);

	/*
	Locks the allocation; error if it's evicted
	*/
	void lock(std::shared_ptr<Allocation> allocation);
	void unlock(std::shared_ptr<Allocation> allocation);

	/*
	Alloc will also lock the allocation immediately
	*/
	std::shared_ptr<Allocation> alloc(unsigned n_pages, std::function<void(void)> eviction_callback);
	void free(std::shared_ptr<Allocation> allocation);

	// Reclaim back all pages
	void clear();

};

class CUDAPageCache final : public PageCache {
private:
	std::vector<char*> baseptrs;
public:
	unsigned gpu_id;
	CUDAPageCache(std::vector<std::pair<char*, uint64_t>> baseptrs,
		uint64_t total_size, uint64_t page_size, const bool allowEvictions,
		unsigned gpu_id);
	~CUDAPageCache() override;
};

}

#endif