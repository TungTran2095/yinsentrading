import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { toast } from 'react-toastify';
import api from '../services/api';

// Async thunks
export const fetchModels = createAsyncThunk(
  'models/fetchModels',
  async (_, { rejectWithValue }) => {
    try {
      const ensembleResponse = await api.get('/models/ensembles');
      const rlResponse = await api.get('/rl/agents');
      
      return {
        ensembles: ensembleResponse.data,
        agents: rlResponse.data
      };
    } catch (error) {
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const trainEnsembleModel = createAsyncThunk(
  'models/trainEnsembleModel',
  async (modelConfig, { rejectWithValue }) => {
    try {
      const response = await api.post('/models/ensembles/train', modelConfig);
      toast.success('Ensemble model training started');
      return response.data;
    } catch (error) {
      toast.error(`Failed to train ensemble model: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const trainRLAgent = createAsyncThunk(
  'models/trainRLAgent',
  async (agentConfig, { rejectWithValue }) => {
    try {
      const response = await api.post('/rl/agents/train', agentConfig);
      toast.success('RL agent training started');
      return response.data;
    } catch (error) {
      toast.error(`Failed to train RL agent: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

// Slice
const modelsSlice = createSlice({
  name: 'models',
  initialState: {
    ensembles: [],
    agents: [],
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      // fetchModels
      .addCase(fetchModels.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchModels.fulfilled, (state, action) => {
        state.ensembles = action.payload.ensembles;
        state.agents = action.payload.agents;
        state.loading = false;
      })
      .addCase(fetchModels.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to fetch models' };
      })
      
      // trainEnsembleModel
      .addCase(trainEnsembleModel.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(trainEnsembleModel.fulfilled, (state, action) => {
        state.ensembles.push(action.payload);
        state.loading = false;
      })
      .addCase(trainEnsembleModel.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to train ensemble model' };
      })
      
      // trainRLAgent
      .addCase(trainRLAgent.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(trainRLAgent.fulfilled, (state, action) => {
        state.agents.push(action.payload);
        state.loading = false;
      })
      .addCase(trainRLAgent.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to train RL agent' };
      });
  },
});

export default modelsSlice.reducer;
