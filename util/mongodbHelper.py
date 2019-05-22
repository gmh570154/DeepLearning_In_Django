import pymongo

class mongoDB:
    def __init__(self,ip = None, port = None, dbName = None):  # 手机开了；连接初始化
        if ip == None and port == None:
            self.myclient = pymongo.MongoClient('mongodb://localhost:27017/')
        else:
            self.myclient = pymongo.MongoClient('mongodb://' + ip + ':'+port+'/')
        if dbName is None:
            self.conn = self.myclient['mycollection']
        else:
            self.conn = self.myclient[dbName]



    def getlistcollections(self):  # 获取数据库下的集合列表
        collist = self.conn.list_collection_names()
        return collist

    def insert_one(self, col, dict_item):  # 插入单个文档
        result = None
        col = self.conn[col]
        result = col.insert(dict_item)
        return result

    def insert_many(self, col, dict_item_list):  # 掺入多个文档
        result = None
        col = self.conn[col]
        result = col.insert_many(dict_item_list)
        return result

    def find_one(self, col):  # 获取集合中的第一个文档
        result = None
        col = self.conn[col]
        result = col.find_one()
        return result

    def find_all(self, col):  # 查询集合中的所有文档
        result = None
        col = self.conn[col]
        result = col.find()
        return result

    def find_by_query(self, col, query):  # 查询多个文档
        result = None
        col = self.conn[col]
        result = col.find(query)
        return result

    def update_one(self, col, query, newvalues):  # 更新单个文档
        result = None
        col = self.conn[col]
        result = col.update_one(query, newvalues)
        return result

    def update_many(self, col, query, newvalues):  # 更新多个文档
        result = None
        col = self.conn[col]
        result = col.update_many(query, newvalues)
        return result

    def delete_one(self, col, query):  # 删除单个文档
        result = None
        col = self.conn[col]
        result = col.delete_one(query)
        return result

    def delete_many(self, col, query):  # 删除多个文档
        result = None
        col = self.conn[col]
        result = col.delete_many(query)
        return result

    def delete_all(self, col):  # 清空集合
        result = None
        col = self.conn[col]
        result = col.delete_many({})
        return result

    def drop_collection(self, col):   # 删除集合
        result = None
        col = self.conn[col]
        result = col.drop()
        return result

    def close_conn(self):
        self.myclient.close()