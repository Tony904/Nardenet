#ifndef LIST_H
#define LIST_H


#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

	typedef struct node node;
	typedef struct list list;

	void list_append(list* lst, void* item);
	void list_append_calloc(list* lst, void* item);
	void* list_get_item(list* lst, size_t index);
	void free_list(list* lst, int free_vals);

	typedef struct node {
		void* val;
		node* next;
		node* prev;
	} node;

	typedef struct list {
		size_t length;
		node* first;
		node* last;
	} list;


#ifdef __cplusplus
}
#endif
#endif