#include "list.h"

#include "xallocs.h"



node* new_node(void* item) {
	node* n = (node*)xmalloc(sizeof(node));
	n->val = item;
	n->next = NULL;
	n->prev = NULL;
	return n;
}

list* new_list() {
	list* lst = (list*)xmalloc(sizeof(list));
	lst->size = 0;
	lst->first = NULL;
	lst->last = NULL;
	return lst;
}

void list_append(list* lst, void* item) {
	node* n = new_node(item);
	if (lst->size == 0) {
		lst->first = n;
		lst->last = n;
		lst->size++;
		return 0;
	}
	lst->last->next = n;
	n->prev = lst->last;
	lst->last = n;
	lst->size++;
}

void* list_get_item(list* lst, size_t index) {
	if (!(lst->size > index)) return NULL;
	node* n = lst->first;
	for (size_t i = 0; i < index; i++) {
		n = n->next;
	}
	return n->val;
}

void free_list(list* lst) {
	node* n = lst->first;
	node* next;
	while (n) {
		next = n->next;
		xfree(n);
		n = next;
	}
	xfree(lst);
}